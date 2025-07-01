"""
Anthropic batch provider implementation using enhanced base classes.

Simplified by delegating common operations to base classes while handling
Anthropic-specific requirements like custom_id hashing and tool schema conversion.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

import anthropic
from rich import print as _print

from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData
from .base import BaseBatchProvider, BatchProviderConfig, ChunkingStrategy
from ..services.file_operations import FileOperationsService


class AnthropicBatchProvider(BaseBatchProvider):
    """Anthropic-specific batch API implementation using enhanced base classes."""

    def __init__(self, file_ops: FileOperationsService, api_key: Optional[str] = None, 
                 dataset_name: Optional[str] = None):
        """
        Initialize AnthropicBatchProvider.

        Args:
            file_ops: File operations service
            api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY environment variable.
            dataset_name: Name of the dataset for unique file naming.
        """
        # Create configuration
        config = BatchProviderConfig(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY'),
            dataset_name=dataset_name
        )

        # Initialize base class
        super().__init__(config, file_ops)

        if not self.config.api_key:
            _print("[bold yellow]Warning: No Anthropic API key configured. Batch submission will fail.")
            _print("[bold yellow]Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.config.api_key) if self.config.api_key else None

        # Store custom_id mappings for Anthropic (required due to hashing)
        self._custom_id_mappings: Dict[str, str] = {}  # hashed_id -> original_id

    def create_batch_request(self, data: ExperimentData, engine_params: EngineParams,
                           raw_messages: List[Dict[str, Any]], custom_id: str,
                           tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create Anthropic batch request object."""
        # Build the params object
        params = {
            "model": engine_params.model,
            "max_tokens": getattr(engine_params, 'max_tokens', 1000),
            "messages": raw_messages
        }

        # Add optional parameters
        if hasattr(engine_params, 'temperature'):
            params["temperature"] = engine_params.temperature

        # Add tools if provided (convert from OpenAI format to Anthropic format)
        if tools:
            anthropic_tools = self._convert_openai_tools_to_anthropic(tools)
            if anthropic_tools:
                params["tools"] = anthropic_tools

        # Anthropic has stricter custom_id requirements (1-64 chars, specific pattern)
        # Generate a shorter, compliant custom_id by hashing the original
        hashed_custom_id = hashlib.md5(custom_id.encode()).hexdigest()[:32]
        
        # Store mapping for later resolution
        self._custom_id_mappings[hashed_custom_id] = custom_id

        # Create the batch request
        batch_request = {
            "custom_id": hashed_custom_id,
            "params": params
        }

        return batch_request

    def get_request_chunks(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Anthropic uses count-based chunking with 10,000 requests per batch."""
        return self.chunk_requests(requests, ChunkingStrategy.BY_COUNT, chunk_size=10000)

    async def _submit_chunk(self, chunk: List[Dict[str, Any]], config_name: str, 
                           chunk_index: int, submission_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Submit a single chunk to Anthropic API."""
        if not self.client:
            _print("[bold red]Anthropic client not configured. Cannot submit batch.")
            return None

        # Generate unique chunk name
        chunk_name = self.generate_unique_batch_name(config_name, chunk_index)

        try:
            _print(f"[bold blue]Submitting Anthropic batch '{chunk_name}' with {len(chunk)} requests...")

            # Submit directly to API (no file upload needed for Anthropic)
            response = self.client.beta.messages.batches.create(requests=chunk)
            
            batch_id = response.id
            _print(f"[bold green]✓ Anthropic batch '{chunk_name}' submitted successfully. Batch ID: {batch_id}")

            # Save custom_id mappings for this chunk
            await self._save_custom_id_mappings(config_name, chunk_index)

            # Update experiment tracking if context provided
            if submission_context:
                await self._update_anthropic_experiment_tracking(
                    config_name, chunk, batch_id, submission_context
                )

            return batch_id

        except Exception as e:
            self.handle_api_error(f"Anthropic batch submission", e, 
                                {"chunk_index": chunk_index, "config": config_name})
            return None

    async def monitor_batches(self, batch_ids: List[str]) -> Dict[str, str]:
        """Monitor Anthropic batch processing status."""
        if not self.client:
            return {batch_id: 'error' for batch_id in batch_ids}

        status_map = {}

        for batch_id in batch_ids:
            try:
                batch = self.client.beta.messages.batches.retrieve(batch_id)
                status_map[batch_id] = batch.processing_status
            except Exception as e:
                self.handle_api_error(f"Anthropic status check", e, {"batch_id": batch_id})
                status_map[batch_id] = 'error'

        return status_map

    async def download_batch_results(self, batch_id: str, output_path: Path) -> Optional[str]:
        """Download batch results for a completed Anthropic batch."""
        if not self.client:
            _print("[bold red]Anthropic client not configured. Cannot download batch results.")
            return None

        try:
            _print(f"[bold blue]Downloading Anthropic results for batch {batch_id}...")

            # Get the batch results
            results = self.client.beta.messages.batches.results(batch_id)
            
            # Write results to file in JSONL format
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                for result in results:
                    # Convert result to dict if needed
                    if hasattr(result, 'model_dump'):
                        result_dict = result.model_dump()
                    elif hasattr(result, '__dict__'):
                        result_dict = result.__dict__
                    else:
                        result_dict = result
                    
                    f.write(json.dumps(result_dict) + '\n')

            _print(f"[bold green]✓ Downloaded Anthropic results to {output_path}")
            return str(output_path)

        except Exception as e:
            self.handle_api_error(f"Anthropic batch download", e, {"batch_id": batch_id})
            return None

    def parse_tool_calls_from_response(self, response_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse tool calls from Anthropic batch response."""
        tool_calls = []

        # Anthropic format: content contains tool_use blocks
        content = response_body.get('content', [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'tool_use':
                    tool_calls.append({
                        'id': block.get('id'),
                        'function': {
                            'name': block.get('name'),
                            'arguments': json.dumps(block.get('input', {}))
                        }
                    })

        return tool_calls

    def extract_response_content(self, response_body: Dict[str, Any]) -> str:
        """Extract response content from Anthropic response."""
        content = response_body.get('content', [])
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
            return '\n'.join(text_parts)
        return str(content)

    def is_response_successful(self, result: Dict[str, Any]) -> bool:
        """Check if Anthropic batch response is successful."""
        # Check if result has error
        if 'error' in result:
            return False
        
        # Check if result has valid response
        if 'result' in result:
            response = result['result']
            return response.get('type') == 'message'
        
        return True

    def get_error_message(self, result: Dict[str, Any]) -> str:
        """Extract error message from Anthropic batch response."""
        error = result.get('error', {})
        if isinstance(error, dict):
            return error.get('message', 'Unknown error')
        return str(error)

    def get_response_body_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the response body from a single batch result item."""
        return result.get('result', result)

    # Private helper methods

    def _convert_openai_tools_to_anthropic(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool schema format to Anthropic tool format."""
        anthropic_tools = []

        for tool in tools:
            if tool.get('type') != 'function':
                continue

            function_def = tool.get('function', {})
            anthropic_tool = {
                'name': function_def.get('name'),
                'description': function_def.get('description'),
                'input_schema': function_def.get('parameters', {})
            }

            # Ensure input_schema has required structure
            if 'input_schema' in anthropic_tool:
                schema = anthropic_tool['input_schema']
                if 'type' not in schema:
                    schema['type'] = 'object'
                if 'properties' not in schema:
                    schema['properties'] = {}

            anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    async def _save_custom_id_mappings(self, config_name: str, chunk_index: int):
        """Save custom_id mappings for this chunk."""
        if not self._custom_id_mappings:
            return

        mapping_file = self.file_ops.directories.batch_metadata / f"{config_name}_anthropic_custom_id_mapping_{chunk_index}.json"
        
        try:
            self.file_ops.write_json_file(self._custom_id_mappings, mapping_file)
            _print(f"[bold blue]Saved {len(self._custom_id_mappings)} custom_id mappings to {mapping_file}")
        except Exception as e:
            self.handle_api_error(f"custom_id mapping save", e, {"config": config_name, "chunk": chunk_index})

    async def _update_anthropic_experiment_tracking(self, config_name: str, chunk: List[Dict[str, Any]], 
                                                   batch_id: str, submission_context: Dict[str, Any]):
        """Update experiment tracking with Anthropic-specific logic."""
        # Extract experiment IDs from the requests in this chunk
        experiment_ids = []
        
        for request in chunk:
            hashed_custom_id = request.get('custom_id', '')
            original_custom_id = self._custom_id_mappings.get(hashed_custom_id, '')
            
            if original_custom_id and '|' in original_custom_id:
                # Parse experiment_id from original custom_id
                parts = original_custom_id.split('|')
                if len(parts) >= 3:
                    experiment_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    experiment_ids.append(experiment_id)

        if experiment_ids and submission_context:
            # Update tracking through the provided context
            tracking_service = submission_context.get('batch_runtime')
            if tracking_service and hasattr(tracking_service, 'tracking_service'):
                await tracking_service.tracking_service.mark_experiments_submitted(
                    config_name, experiment_ids, [batch_id]
                )