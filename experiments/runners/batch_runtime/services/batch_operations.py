"""
Service to orchestrate batch operations across all providers.

This service coordinates batch submission, monitoring, and result processing
across different AI providers (OpenAI, Anthropic, Gemini).
"""

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable

from rich import print as _print

from agent.src.logger import create_logger
from agent.src.shopper import SimulatedShopper
from agent.src.typedefs import EngineParams
from agent.src.core.tools import AddToCartInput
from experiments.filesystem_environment import FilesystemShoppingEnvironment
from experiments.config import ExperimentData
from experiments.data_loader import experiments_iter
from experiments.results import aggregate_model_data

from .file_operations import FileOperationsService
from .experiment_tracking import ExperimentTrackingService


class BatchOperationsService:
    """Service to orchestrate batch operations across all providers."""
    
    def __init__(self, providers: Dict[str, Any], tracking_service: ExperimentTrackingService,
                 file_ops: FileOperationsService):
        self.providers = providers
        self.tracking_service = tracking_service
        self.file_ops = file_ops
    
    async def submit_all_batches(self, supported_engines: List[EngineParams], 
                               experiments_df, run_output_dir: Path, 
                               force_submit: bool = False) -> Dict[str, List[str]]:
        """Submit batches for all supported engines."""
        all_submitted_batches = {}
        
        # Submit batches for each engine in parallel
        submission_tasks = []
        for engine_params in supported_engines:
            task = asyncio.create_task(
                self._submit_batches_for_engine(engine_params, experiments_df, 
                                              run_output_dir, force_submit)
            )
            submission_tasks.append((engine_params.config_name, task))
        
        # Wait for all submissions to complete
        for config_name, task in submission_tasks:
            try:
                batch_ids = await task
                if batch_ids:
                    all_submitted_batches[config_name] = batch_ids
                    _print(f"[bold green]✓ Submitted {len(batch_ids)} batches for {config_name}")
                else:
                    _print(f"[bold yellow]No batches submitted for {config_name}")
            except Exception as e:
                _print(f"[bold red]Failed to submit batches for {config_name}: {e}")
        
        return all_submitted_batches
    
    async def monitor_and_process_results(self, batch_mapping: Dict[str, List[str]], 
                                        run_output_dir: Path, experiments_df):
        """Monitor batches and process results as they complete."""
        if not batch_mapping:
            _print("[bold yellow]No batches to monitor")
            return
        
        _print(f"[bold blue]Monitoring {sum(len(batches) for batches in batch_mapping.values())} batches across {len(batch_mapping)} providers...")
        
        # Start monitoring tasks for each provider
        monitoring_tasks = []
        for config_name, batch_ids in batch_mapping.items():
            if batch_ids:
                provider = self.providers.get(config_name)
                if provider:
                    task = asyncio.create_task(
                        self._monitor_provider_batches(provider, batch_ids, config_name, 
                                                     run_output_dir, experiments_df)
                    )
                    monitoring_tasks.append(task)
        
        # Wait for all monitoring to complete
        if monitoring_tasks:
            await asyncio.gather(*monitoring_tasks, return_exceptions=True)
        
        _print("[bold green]✓ All batch monitoring completed")
    
    async def _submit_batches_for_engine(self, engine_params: EngineParams, experiments_df,
                                       run_output_dir: Path, force_submit: bool) -> List[str]:
        """Submit batches for a single engine."""
        config_name = engine_params.config_name
        provider = self.providers.get(config_name)
        
        if not provider:
            _print(f"[bold red]No provider found for {config_name}")
            return []
        
        # Get outstanding experiments
        all_experiments = list(experiments_iter(experiments_df))
        outstanding_experiments = await self.tracking_service.get_outstanding_experiments(
            config_name, all_experiments, experiments_df, force_submit
        )
        
        if not outstanding_experiments:
            _print(f"[bold green]All experiments already completed for {config_name}")
            return []
        
        # Generate batch requests
        batch_requests = []
        experiments_in_batch = []
        
        for data in outstanding_experiments:
            try:
                request = self._generate_batch_request(data, engine_params)
                batch_requests.append(request)
                experiments_in_batch.append(data.experiment_id)
            except Exception as e:
                _print(f"[bold red]Error generating request for {data.experiment_id}: {e}")
        
        if not batch_requests:
            return []
        
        # Submit to provider
        _print(f"[bold blue]Submitting {len(batch_requests)} requests for {config_name}...")
        
        submission_context = {
            'experiments_to_run': outstanding_experiments,
            'engine_params': engine_params,
            'batch_runtime': self,
            'experiment_ids': experiments_in_batch
        }
        
        submitted_ids = await provider.upload_and_submit_batches(
            batch_requests, config_name, run_output_dir, submission_context
        )
        
        # Update tracking
        if submitted_ids and experiments_in_batch:
            await self.tracking_service.mark_experiments_submitted(
                config_name, experiments_in_batch, submitted_ids
            )
        
        return submitted_ids
    
    async def _monitor_provider_batches(self, provider, batch_ids: List[str], config_name: str,
                                      run_output_dir: Path, experiments_df):
        """Monitor batches for a specific provider."""
        _print(f"[bold blue]Monitoring {len(batch_ids)} batches for {config_name}...")
        
        completed_batches = set()
        check_interval = 10
        
        while True:
            try:
                # Check batch statuses
                status_map = await provider.monitor_batches(batch_ids)
                
                if not status_map:
                    _print(f"[bold yellow]No status received for {config_name} batches")
                    await asyncio.sleep(check_interval)
                    continue
                
                # Process newly completed batches
                newly_completed = []
                for batch_id, status in status_map.items():
                    if status in ['completed', 'ended'] and batch_id not in completed_batches:
                        completed_batches.add(batch_id)
                        newly_completed.append(batch_id)
                
                # Download and process results for completed batches
                if newly_completed:
                    for batch_id in newly_completed:
                        await self._process_completed_batch(provider, batch_id, config_name, 
                                                          run_output_dir, experiments_df)
                
                # Check if all batches are done
                completed_count = sum(1 for status in status_map.values() 
                                    if status in ['completed', 'ended'])
                failed_count = sum(1 for status in status_map.values() 
                                 if status in ['failed', 'error', 'cancelled'])
                in_progress_count = len(status_map) - completed_count - failed_count
                
                _print(f"[bold cyan]{config_name} status: {completed_count} completed, {in_progress_count} in progress, {failed_count} failed")
                
                if in_progress_count == 0:
                    _print(f"[bold green]All batches completed for {config_name}")
                    # Run final aggregation
                    await self._run_final_aggregation_for_provider(config_name, run_output_dir, experiments_df)
                    break
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                _print(f"[bold red]Error monitoring {config_name}: {e}")
                await asyncio.sleep(check_interval)
    
    async def _process_completed_batch(self, provider, batch_id: str, config_name: str,
                                     run_output_dir: Path, experiments_df):
        """Process a completed batch by downloading and processing results."""
        try:
            _print(f"[bold cyan]Processing completed batch {batch_id} for {config_name}...")
            
            # Download results
            results_file = self.file_ops.get_batch_results_file(config_name, batch_id)
            downloaded_path = await provider.download_batch_results(batch_id, results_file)
            
            if downloaded_path:
                # Process the results file
                await self._process_batch_results_file(downloaded_path, config_name, 
                                                     provider, run_output_dir, experiments_df)
                _print(f"[bold green]✓ Processed batch {batch_id} for {config_name}")
            else:
                _print(f"[bold red]Failed to download results for batch {batch_id}")
                
        except Exception as e:
            _print(f"[bold red]Error processing batch {batch_id}: {e}")
    
    async def _process_batch_results_file(self, results_file_path: str, config_name: str,
                                        provider, run_output_dir: Path, experiments_df):
        """Process a batch results file and create experiment outputs."""
        results = self.file_ops.read_jsonl_file(Path(results_file_path))
        
        if not results:
            _print(f"[bold yellow]No results found in {results_file_path}")
            return
        
        _print(f"[bold blue]Processing {len(results)} results from {results_file_path}")
        
        # Process each result
        for result in results:
            try:
                await self._process_single_result(result, provider, config_name, 
                                                run_output_dir, experiments_df)
            except Exception as e:
                _print(f"[bold red]Error processing single result: {e}")
    
    async def _process_single_result(self, result: Dict[str, Any], provider, config_name: str,
                                   run_output_dir: Path, experiments_df):
        """Process a single batch result and create experiment output."""
        # Get custom_id and parse experiment info
        custom_id = result.get('custom_id')
        if not custom_id:
            return
        
        # Parse custom_id to get experiment info
        if '|' not in custom_id:
            return
        
        parts = custom_id.split('|')
        if len(parts) < 4:
            return
        
        query, experiment_label, experiment_number, _ = parts[:4]
        
        # Find corresponding experiment data
        experiment_data = None
        for data in experiments_iter(experiments_df):
            if (data.query == query and 
                data.experiment_label == experiment_label and 
                str(data.experiment_number) == experiment_number):
                experiment_data = data
                break
        
        if not experiment_data:
            _print(f"[bold yellow]Could not find experiment data for {custom_id}")
            return
        
        # Check if response was successful
        if not provider.is_response_successful(result):
            error_msg = provider.get_error_message(result)
            _print(f"[bold red]Batch request failed for {custom_id}: {error_msg}")
            return
        
        # Extract response and tool calls
        response_body = provider.get_response_body_from_result(result)
        tool_calls = provider.parse_tool_calls_from_response(response_body)
        
        # Find add_to_cart tool call
        add_to_cart_call = None
        for tool_call in tool_calls:
            if tool_call.get('function', {}).get('name') == 'add_to_cart':
                add_to_cart_call = tool_call
                break
        
        if not add_to_cart_call:
            _print(f"[bold yellow]No add_to_cart tool call found for {custom_id}")
            return
        
        # Parse tool call arguments
        try:
            args_str = add_to_cart_call['function'].get('arguments', '{}')
            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
            add_to_cart_input = AddToCartInput.model_validate(args_dict)
        except Exception as e:
            _print(f"[bold red]Failed to parse add_to_cart arguments for {custom_id}: {e}")
            return
        
        # Create experiment output
        await self._create_experiment_output(experiment_data, config_name, add_to_cart_input, 
                                           response_body, provider, run_output_dir)
    
    async def _create_experiment_output(self, data: ExperimentData, config_name: str,
                                      add_to_cart_input: AddToCartInput, response_body: Dict[str, Any],
                                      provider, run_output_dir: Path):
        """Create experiment output directory and files."""
        try:
            # Find engine params for this config
            engine_params = None
            for ep in self.providers.keys():
                if ep == config_name:
                    # We'd need to store engine params mapping - for now use a simple approach
                    from agent.src.typedefs import EngineParams
                    engine_params = EngineParams(
                        engine_type="openai",  # This would need to be determined properly
                        model="gpt-4",
                        config_name=config_name
                    )
                    break
            
            if not engine_params:
                _print(f"[bold yellow]Could not find engine params for {config_name}")
                return
            
            experiment_df = data.experiment_df
            model_output_dir = data.model_output_dir(run_output_dir, engine_params)
            
            # Create logger and record experiment results
            with create_logger(
                data.query,
                output_dir=model_output_dir,
                experiment_df=experiment_df,
                engine_params=engine_params,
                experiment_label=data.experiment_label,
                experiment_number=data.experiment_number,
                silent=True
            ) as logger:
                # Record the cart item
                logger.record_cart_item(add_to_cart_input)
                
                # Create mock AIMessage for the response
                from langchain_core.messages import AIMessage
                tool_call_dict = {
                    'name': 'add_to_cart',
                    'args': add_to_cart_input.model_dump(),
                    'id': 'batch_result'
                }
                
                ai_message = AIMessage(
                    content=provider.extract_response_content(response_body),
                    tool_calls=[tool_call_dict]
                )
                
                # Record the agent interaction
                logger.record_agent_interaction(ai_message)
                
        except Exception as e:
            _print(f"[bold red]Error creating experiment output for {data.experiment_id}: {e}")
    
    async def _run_final_aggregation_for_provider(self, config_name: str, run_output_dir: Path, experiments_df):
        """Run final aggregation for a provider."""
        try:
            _print(f"[bold blue]Running final aggregation for {config_name}...")
            
            # Find model directories to aggregate
            model_dirs_to_aggregate = set()
            
            for data in experiments_iter(experiments_df):
                # Check if experiment was completed
                # We'd need engine_params here - simplified for now
                journey_dir = run_output_dir / config_name / data.query / data.experiment_label / f"{data.experiment_label}_{data.experiment_number}"
                experiment_data_file = journey_dir / "experiment_data.csv"
                
                if experiment_data_file.exists():
                    model_output_dir = journey_dir.parent.parent.parent
                    model_dirs_to_aggregate.add(model_output_dir)
            
            # Run aggregation for each model directory
            for model_dir in model_dirs_to_aggregate:
                try:
                    aggregate_model_data(model_dir)
                except Exception as e:
                    _print(f"[bold red]Error aggregating {model_dir}: {e}")
            
            _print(f"[bold green]✓ Final aggregation completed for {config_name}")
            
        except Exception as e:
            _print(f"[bold red]Error during final aggregation for {config_name}: {e}")
    
    def _generate_batch_request(self, data: ExperimentData, engine_params: EngineParams) -> dict:
        """Generate a batch request for the given experiment."""
        # Create filesystem environment
        screenshots_dir = Path("local_datasets/screenshots") / data.query.replace(" ", "_")
        environment = FilesystemShoppingEnvironment(
            screenshots_dir=screenshots_dir,
            query=data.query,
            experiment_label=data.experiment_label,
            experiment_number=data.experiment_number,
            remote=engine_params.engine_type.lower() == "gemini"
        )
        
        # Create temporary shopper to generate request
        shopper = SimulatedShopper(
            initial_message=data.prompt_template or "",
            engine_params=engine_params,
            environment=environment,
            logger=None,
        )
        
        # Get raw message requests
        raw_messages = shopper.get_batch_request()
        
        # Create custom_id
        custom_id = f"{data.query}|{data.experiment_label}|{data.experiment_number}|{engine_params.config_name}"
        
        # Extract tool definitions
        tools = []
        for tool in shopper.agent.tools:
            if hasattr(tool, 'args_schema') and tool.args_schema:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.args_schema.model_json_schema()
                    }
                }
                tools.append(tool_def)
        
        # Use provider to create request
        provider = self.providers.get(engine_params.config_name)
        if provider:
            return provider.create_batch_request(data, engine_params, raw_messages, custom_id, tools)
        else:
            raise ValueError(f"No provider found for {engine_params.config_name}")