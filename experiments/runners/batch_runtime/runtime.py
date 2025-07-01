"""
Simplified BatchEvaluationRuntime using composed services.

This runtime orchestrates batch processing by delegating responsibilities
to specialized services rather than handling everything internally.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable

from rich import print as _print

from agent.src.logger import create_logger
from agent.src.shopper import SimulatedShopper
from agent.src.typedefs import EngineParams
from agent.src.core.tools import AddToCartInput
from experiments.filesystem_environment import FilesystemShoppingEnvironment
from experiments.config import ExperimentData
from experiments.data_loader import experiments_iter, load_experiment_data
from experiments.runners.simple_runtime import BaseEvaluationRuntime
from experiments.results import aggregate_model_data

from .services.file_operations import FileOperationsService
from .services.experiment_tracking import ExperimentTrackingService
from .services.batch_operations import BatchOperationsService
from ..services.screenshot_validation import ScreenshotValidationService
from .providers.base import BatchProviderConfig
from .providers.openai import OpenAIBatchProvider
from .providers.anthropic import AnthropicBatchProvider
from .providers.gemini import GeminiBatchProvider




class BatchEvaluationRuntime(BaseEvaluationRuntime):
    """
    Simplified batch evaluation runtime using composed services.
    
    This runtime orchestrates batch processing by delegating to specialized services
    rather than handling all operations internally.
    """
    
    def __init__(
        self, 
        local_dataset_path: str,
        engine_params_list: List[EngineParams], 
        output_dir_override: Optional[str] = None,
        experiment_count_limit: Optional[int] = None,
        experiment_label_filter: Optional[str] = None,
        debug_mode: bool = False,
        force_submit: bool = False
    ):
        """Initialize the simplified BatchEvaluationRuntime."""
        # Extract dataset name from path
        dataset_filename = os.path.splitext(os.path.basename(local_dataset_path))[0]
        super().__init__(dataset_filename, output_dir_override, debug_mode)
        
        # Store configuration
        self.local_dataset_path = local_dataset_path
        self.engine_params_list = engine_params_list
        self.experiment_count_limit = experiment_count_limit
        self.experiment_label_filter = experiment_label_filter
        self.force_submit = force_submit
        
        # Load dataset
        self.experiments_df = load_experiment_data(local_dataset_path)
        
        # Initialize services
        self.file_ops = FileOperationsService(self.run_output_dir)
        self.tracking_service = ExperimentTrackingService(self.run_output_dir)
        
        # Set up screenshots directory
        dataset_dir = Path(local_dataset_path).parent
        self.screenshots_dir = dataset_dir / "screenshots" / self.dataset_name
        self.screenshot_service = ScreenshotValidationService(self.screenshots_dir)
        
        # Create providers
        self.providers = self._create_providers()
        
        # Create batch operations service
        self.batch_service = BatchOperationsService(
            providers=self.providers,
            tracking_service=self.tracking_service,
            file_ops=self.file_ops
        )
        
        # Filter supported engines
        self.supported_engines = [ep for ep in engine_params_list if ep.config_name in self.providers]
        
        if not self.supported_engines:
            raise ValueError("No supported engines found for batch processing")
        
        _print(f"[bold green]Initialized BatchEvaluationRuntime with {len(self.supported_engines)} supported engines")
    
    @property
    def experiments_iter(self) -> Iterable[ExperimentData]:
        """Return an iterator over experiments from the local dataset."""
        return experiments_iter(self.experiments_df)
    
    def _create_providers(self) -> Dict[str, Any]:
        """Create batch providers for supported engine types."""
        providers = {}
        
        for engine in self.engine_params_list:
            engine_type = engine.engine_type.lower()
            config_name = engine.config_name
            
            try:
                if engine_type == "openai":
                    providers[config_name] = OpenAIBatchProvider(
                        file_ops=self.file_ops,
                        dataset_name=self.dataset_name
                    )
                elif engine_type == "anthropic":
                    providers[config_name] = AnthropicBatchProvider(
                        file_ops=self.file_ops,
                        dataset_name=self.dataset_name
                    )
                elif engine_type == "gemini":
                    providers[config_name] = GeminiBatchProvider(
                        file_ops=self.file_ops,
                        screenshots_dir=self.screenshots_dir,
                        dataset_name=self.dataset_name
                    )
                else:
                    _print(f"[bold yellow]Unsupported engine type: {engine_type}")
                    
            except Exception as e:
                _print(f"[bold yellow]Failed to create provider for {config_name}: {e}")
        
        return providers
    
    async def run(self):
        """
        Main entry point for simplified batch processing.
        
        This method orchestrates the entire batch processing workflow using services.
        """
        # Validate screenshots
        if not self.screenshot_service.validate_all_screenshots(self.experiments_df, self.local_dataset_path):
            raise RuntimeError("Screenshot validation failed. Cannot proceed with batch processing.")
        
        _print(f"[bold blue]Starting batch processing for {len(self.supported_engines)} engines...")
        _print(f"[bold blue]Output directory: {self.run_output_dir}")
        
        if self.force_submit:
            _print(f"[bold yellow]Force submit mode enabled")
        
        # Submit all batches
        batch_mapping = await self.batch_service.submit_all_batches(
            self.supported_engines, self.experiments_df, self.run_output_dir, self.force_submit
        )
        
        if not batch_mapping:
            _print("[bold yellow]No batches were submitted")
            return
        
        # Monitor and process results
        await self.batch_service.monitor_and_process_results(
            batch_mapping, self.run_output_dir, self.experiments_df
        )
        
        _print("[bold green]✓ Batch processing completed successfully!")