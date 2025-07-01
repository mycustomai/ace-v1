"""
Gemini batch provider implementation using enhanced base classes.

Simplified by delegating common operations to base classes while handling
Gemini-specific requirements like GCS integration and tool schema conversion.
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich import print as _print

from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData
from .base import BaseBatchProvider, BatchProviderConfig, ChunkingStrategy
from ..services.file_operations import FileOperationsService


class GeminiBatchProvider(BaseBatchProvider):
    """Gemini-specific batch API implementation using enhanced base classes."""

    def __init__(self, file_ops: FileOperationsService, screenshots_dir: Optional[Path] = None,
                 project_id: Optional[str] = None, location: str = "us-central1",
                 bucket_name: Optional[str] = None, dataset_name: Optional[str] = None):
        """
        Initialize GeminiBatchProvider.

        Args:
            file_ops: File operations service
            screenshots_dir: Directory containing screenshots for upload
            project_id: Google Cloud project ID
            location: Google Cloud location
            bucket_name: GCS bucket name for screenshots
            dataset_name: Name of the dataset for unique file naming
        """
        # Create configuration
        config = BatchProviderConfig(
            api_key=os.getenv('GOOGLE_API_KEY'),
            project_id=project_id or os.getenv('GOOGLE_CLOUD_PROJECT'),
            location=location,
            bucket_name=bucket_name or os.getenv('GCS_BUCKET_NAME'),
            dataset_name=dataset_name
        )

        # Initialize base class
        super().__init__(config, file_ops)

        self.screenshots_dir = Path(screenshots_dir) if screenshots_dir else None
        
        # Validate Google Cloud configuration
        errors = self._validate_google_cloud_config()
        if errors:
            _print(f"[bold yellow]Google Cloud configuration warnings: {', '.join(errors)}")

        # Initialize clients lazily
        self._batch_client = None
        self._storage_client = None
        self._uploaded_screenshots: Dict[str, str] = {}  # local_path -> gcs_url

    def create_batch_request(self, data: ExperimentData, engine_params: EngineParams,
                           raw_messages: List[Dict[str, Any]], custom_id: str,
                           tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create Gemini batch request object."""
        # Convert messages to Gemini format
        gemini_messages = self._convert_messages_to_gemini_format(raw_messages)
        
        # Build the request
        batch_request = {
            "custom_id": custom_id,
            "method": "POST", 
            "uri": f"https://generativelanguage.googleapis.com/v1beta/models/{engine_params.model}:generateContent",
            "body": {
                "contents": gemini_messages,
                "generationConfig": {
                    "maxOutputTokens": getattr(engine_params, 'max_tokens', 1000),
                    "temperature": getattr(engine_params, 'temperature', 0.7),
                }
            }
        }

        # Add tools if provided (convert from OpenAI format to Gemini format)
        if tools:
            gemini_tools = self._convert_openai_tools_to_gemini(tools)
            if gemini_tools:
                batch_request["body"]["tools"] = gemini_tools

        return batch_request

    def get_request_chunks(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Gemini uses count-based chunking with 1,000 requests per batch."""
        return self.chunk_requests(requests, ChunkingStrategy.BY_COUNT, chunk_size=1000)

    async def _submit_chunk(self, chunk: List[Dict[str, Any]], config_name: str, 
                           chunk_index: int, submission_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Submit a single chunk to Gemini API."""
        try:
            # Ensure batch client is initialized
            if not await self._ensure_batch_client():
                return None

            # Generate unique chunk name
            chunk_name = self.generate_unique_batch_name(config_name, chunk_index)

            _print(f"[bold blue]Submitting Gemini batch '{chunk_name}' with {len(chunk)} requests...")

            # Write chunk to input file
            input_file = self.write_batch_input_file(chunk, config_name, f"chunk_{chunk_index}")

            # Submit to API
            response = await self._submit_batch_to_gemini_api(input_file, chunk_name)
            
            if response and 'name' in response:
                batch_id = response['name']
                _print(f"[bold green]✓ Gemini batch '{chunk_name}' submitted successfully. Batch ID: {batch_id}")
                return batch_id
            else:
                _print(f"[bold red]✗ Failed to submit Gemini batch '{chunk_name}': Invalid response")
                return None

        except Exception as e:
            self.handle_api_error(f"Gemini batch submission", e, 
                                {"chunk_index": chunk_index, "config": config_name})
            return None

    async def monitor_batches(self, batch_ids: List[str]) -> Dict[str, str]:
        """Monitor Gemini batch processing status."""
        if not await self._ensure_batch_client():
            return {batch_id: 'error' for batch_id in batch_ids}

        status_map = {}

        for batch_id in batch_ids:
            try:
                # Get batch status from API
                status = await self._get_batch_status(batch_id)
                status_map[batch_id] = status
            except Exception as e:
                self.handle_api_error(f"Gemini status check", e, {"batch_id": batch_id})
                status_map[batch_id] = 'error'

        return status_map

    async def download_batch_results(self, batch_id: str, output_path: Path) -> Optional[str]:
        """Download batch results for a completed Gemini batch."""
        if not await self._ensure_batch_client():
            return None

        try:
            _print(f"[bold blue]Downloading Gemini results for batch {batch_id}...")

            # Get batch results from API
            results = await self._download_batch_results_from_api(batch_id)
            
            if not results:
                _print(f"[bold red]No results found for Gemini batch {batch_id}")
                return None

            # Write results to file in JSONL format
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.file_ops.write_jsonl_file(results, output_path)
            
            _print(f"[bold green]✓ Downloaded Gemini results to {output_path}")
            return str(output_path)

        except Exception as e:
            self.handle_api_error(f"Gemini batch download", e, {"batch_id": batch_id})
            return None

    def parse_tool_calls_from_response(self, response_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse tool calls from Gemini batch response."""
        tool_calls = []

        # Gemini format: candidates[0].content.parts contain functionCall blocks
        candidates = response_body.get('candidates', [])
        if candidates:
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            
            for part in parts:
                if 'functionCall' in part:
                    func_call = part['functionCall']
                    tool_calls.append({
                        'function': {
                            'name': func_call.get('name'),
                            'arguments': json.dumps(func_call.get('args', {}))
                        }
                    })

        return tool_calls

    def extract_response_content(self, response_body: Dict[str, Any]) -> str:
        """Extract response content from Gemini response."""
        candidates = response_body.get('candidates', [])
        if candidates:
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            
            text_parts = []
            for part in parts:
                if 'text' in part:
                    text_parts.append(part['text'])
            
            return '\n'.join(text_parts)
        
        return ''

    def is_response_successful(self, result: Dict[str, Any]) -> bool:
        """Check if Gemini batch response is successful."""
        # Check for errors in the result
        if 'error' in result:
            return False
        
        # Check if response has valid candidates
        response = result.get('response', result)
        candidates = response.get('candidates', [])
        
        return len(candidates) > 0

    def get_error_message(self, result: Dict[str, Any]) -> str:
        """Extract error message from Gemini batch response."""
        error = result.get('error', {})
        if isinstance(error, dict):
            return error.get('message', 'Unknown error')
        return str(error)

    def get_response_body_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the response body from a single batch result item."""
        return result.get('response', result)

    # Screenshot management methods

    async def upload_required_screenshots(self, screenshot_paths: List[str]):
        """Upload specific screenshots to GCS."""
        if not await self._ensure_storage_client():
            _print("[bold red]GCS client not available. Cannot upload screenshots.")
            return

        _print(f"[bold blue]Uploading {len(screenshot_paths)} screenshots to GCS...")
        
        for screenshot_path in screenshot_paths:
            try:
                gcs_url = await self._upload_screenshot_to_gcs(screenshot_path)
                if gcs_url:
                    self._uploaded_screenshots[screenshot_path] = gcs_url
            except Exception as e:
                self.handle_api_error(f"screenshot upload", e, {"path": screenshot_path})

    # Private helper methods

    def _validate_google_cloud_config(self) -> List[str]:
        """Validate Google Cloud configuration."""
        errors = []
        
        if not self.config.api_key:
            errors.append("GOOGLE_API_KEY not set")
        if not self.config.project_id:
            errors.append("GOOGLE_CLOUD_PROJECT not set")
        if not self.config.bucket_name:
            errors.append("GCS_BUCKET_NAME not set")
            
        return errors

    async def _ensure_batch_client(self) -> bool:
        """Ensure batch client is initialized."""
        if self._batch_client is None:
            try:
                # Import and initialize Gemini batch client
                # This would require the actual Google AI client library
                _print("[bold yellow]Gemini batch client initialization not implemented")
                return False
            except Exception as e:
                self.handle_api_error("Gemini client initialization", e)
                return False
        return True

    async def _ensure_storage_client(self) -> bool:
        """Ensure GCS storage client is initialized."""
        if self._storage_client is None:
            try:
                # Import and initialize GCS client
                # This would require the google-cloud-storage library
                _print("[bold yellow]GCS client initialization not implemented")
                return False
            except Exception as e:
                self.handle_api_error("GCS client initialization", e)
                return False
        return True

    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI message format to Gemini format."""
        gemini_messages = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Map roles
            if role == 'assistant':
                gemini_role = 'model'
            else:
                gemini_role = 'user'
            
            gemini_message = {
                "role": gemini_role,
                "parts": []
            }
            
            # Handle content (text or multimodal)
            if isinstance(content, str):
                gemini_message["parts"].append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text':
                        gemini_message["parts"].append({"text": part.get('text', '')})
                    elif part.get('type') == 'image_url':
                        # Handle image content - would need to convert to Gemini format
                        image_url = part.get('image_url', {}).get('url', '')
                        if image_url:
                            gemini_message["parts"].append({
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_url  # This would need proper conversion
                                }
                            })
            
            gemini_messages.append(gemini_message)
        
        return gemini_messages

    def _convert_openai_tools_to_gemini(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool schema format to Gemini tool format."""
        gemini_tools = []
        
        for tool in tools:
            if tool.get('type') != 'function':
                continue
            
            function_def = tool.get('function', {})
            gemini_tool = {
                "function_declarations": [{
                    "name": function_def.get('name'),
                    "description": function_def.get('description'),
                    "parameters": function_def.get('parameters', {})
                }]
            }
            
            gemini_tools.append(gemini_tool)
        
        return gemini_tools

    async def _submit_batch_to_gemini_api(self, input_file: Path, chunk_name: str) -> Optional[Dict[str, Any]]:
        """Submit batch to Gemini API."""
        # This would implement the actual API call to Gemini batch service
        _print(f"[bold yellow]Gemini batch API submission not implemented for {chunk_name}")
        return None

    async def _get_batch_status(self, batch_id: str) -> str:
        """Get status of a Gemini batch."""
        # This would implement the actual API call to check batch status
        return 'unknown'

    async def _download_batch_results_from_api(self, batch_id: str) -> Optional[List[Dict[str, Any]]]:
        """Download batch results from Gemini API."""
        # This would implement the actual API call to download results
        return None

    async def _upload_screenshot_to_gcs(self, screenshot_path: str) -> Optional[str]:
        """Upload a screenshot to GCS and return the public URL."""
        # This would implement the actual GCS upload
        return None