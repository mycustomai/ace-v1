"""
Experiment tracking service for batch processing.

Centralizes all experiment status tracking and batch mapping operations.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from rich import print as _print

from experiments.config import ExperimentData
from experiments.data_loader import experiments_iter
from agent.src.typedefs import EngineParams


class ExperimentStatus(Enum):
    """Status of an individual experiment."""
    NOT_SUBMITTED = "not_submitted"
    SUBMITTED = "submitted" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExperimentStatusInfo:
    """Detailed status information for an experiment."""
    experiment_id: str
    status: ExperimentStatus
    batch_id: Optional[str] = None
    submitted_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    
    @property
    def is_outstanding(self) -> bool:
        """True if experiment needs processing (not submitted or not completed)."""
        return self.status in [ExperimentStatus.NOT_SUBMITTED, ExperimentStatus.FAILED]


class ExperimentTrackingService:
    """Centralized experiment status tracking and batch mapping."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.batch_metadata_dir = self.output_dir / "batch_metadata"
        self.batch_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.mapping_file = self.batch_metadata_dir / "experiment_batch_mapping.json"
        self.submitted_experiments_file = self.batch_metadata_dir / "submitted_experiments.json"
        
        # Async lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def get_experiment_status(self, config_name: str, experiments_df) -> Dict[str, ExperimentStatusInfo]:
        """
        Get comprehensive status for all experiments for a given configuration.
        
        Args:
            config_name: Configuration name
            experiments_df: DataFrame containing experiments
            
        Returns:
            Dictionary mapping experiment_id to ExperimentStatusInfo
        """
        async with self._lock:
            # Get all experiments from the dataset
            dataset_experiments = {data.experiment_id for data in experiments_iter(experiments_df)}
            
            # Load submission status from tracking files
            submission_status = self._load_submission_status(config_name, dataset_experiments)
            
            # Check completion status by examining filesystem
            completion_status = self._check_completion_status(config_name, experiments_df)
            
            # Combine into comprehensive status
            comprehensive_status = {}
            
            for exp_id in dataset_experiments:
                submitted = submission_status.get(exp_id, False)
                completed = completion_status.get(exp_id, False)
                
                if completed:
                    status = ExperimentStatus.COMPLETED
                elif submitted:
                    status = ExperimentStatus.SUBMITTED
                else:
                    status = ExperimentStatus.NOT_SUBMITTED
                
                # Get batch_id if available
                batch_id = self._get_experiment_batch_id(config_name, exp_id)
                
                comprehensive_status[exp_id] = ExperimentStatusInfo(
                    experiment_id=exp_id,
                    status=status,
                    batch_id=batch_id
                )
            
            return comprehensive_status
    
    async def mark_experiments_submitted(self, config_name: str, experiment_ids: List[str], 
                                       batch_ids: List[str]) -> None:
        """
        Mark experiments as submitted with thread safety.
        
        Args:
            config_name: Configuration name
            experiment_ids: List of experiment IDs that were submitted
            batch_ids: List of batch IDs they were submitted to
        """
        if not experiment_ids or not batch_ids:
            _print(f"[bold yellow]Warning: Empty experiment IDs or batch IDs for {config_name}")
            return
        
        async with self._lock:
            # Update the detailed experiment-to-batch mapping
            await self._update_experiment_batch_mapping(config_name, experiment_ids, batch_ids)
            
            # Update the simple submitted experiments tracking
            await self._update_submitted_experiments_tracking(config_name, experiment_ids)
    
    async def get_outstanding_experiments(self, config_name: str, all_experiments: List[ExperimentData],
                                        experiments_df, force_submit: bool = False) -> List[ExperimentData]:
        """
        Get list of experiments that need processing.
        
        Args:
            config_name: Configuration name
            all_experiments: List of all experiments
            experiments_df: DataFrame containing experiments
            force_submit: If True, only skip completed experiments
            
        Returns:
            List of experiments that need processing
        """
        status_map = await self.get_experiment_status(config_name, experiments_df)
        outstanding_experiments = []
        
        for experiment in all_experiments:
            exp_status = status_map.get(experiment.experiment_id)
            
            if exp_status is None:
                # Unknown experiment, include it
                outstanding_experiments.append(experiment)
                continue
            
            if force_submit:
                # When force_submit is enabled, only skip completed experiments
                if exp_status.status != ExperimentStatus.COMPLETED:
                    outstanding_experiments.append(experiment)
            else:
                # Normal behavior: include outstanding experiments
                if exp_status.is_outstanding:
                    outstanding_experiments.append(experiment)
        
        return outstanding_experiments
    
    def print_status_report(self, config_name: str, status_map: Dict[str, ExperimentStatusInfo]) -> None:
        """Print a detailed status report for experiments."""
        _print(f"\\n[bold blue]══════ Experiment Status Report for {config_name} ══════")
        
        total = len(status_map)
        submitted = sum(1 for s in status_map.values() if s.status == ExperimentStatus.SUBMITTED)
        completed = sum(1 for s in status_map.values() if s.status == ExperimentStatus.COMPLETED)
        outstanding = sum(1 for s in status_map.values() if s.is_outstanding)
        
        _print(f"[bold green]Total Experiments: {total}")
        _print(f"[bold yellow]Submitted: {submitted}")
        _print(f"[bold green]Completed: {completed}")
        _print(f"[bold red]Outstanding: {outstanding}")
        _print(f"[bold blue]══════ End Status Report ══════\\n")
    
    def _load_submission_status(self, config_name: str, dataset_experiments: Set[str]) -> Dict[str, bool]:
        """Load submission status from tracking files."""
        if not self.mapping_file.exists():
            return {exp_id: False for exp_id in dataset_experiments}
        
        try:
            with open(self.mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            if config_name not in mapping_data:
                return {exp_id: False for exp_id in dataset_experiments}
            
            provider_data = mapping_data[config_name]
            experiment_status = {}
            
            for exp_id in dataset_experiments:
                if exp_id in provider_data.get("experiments", {}):
                    exp_data = provider_data["experiments"][exp_id]
                    # Check if experiment has valid batch_id
                    batch_id = exp_data.get("batch_id")
                    if batch_id and isinstance(batch_id, str) and batch_id.strip():
                        experiment_status[exp_id] = True
                    else:
                        experiment_status[exp_id] = False
                else:
                    experiment_status[exp_id] = False
            
            return experiment_status
            
        except (json.JSONDecodeError, KeyError) as e:
            _print(f"[bold yellow]Warning: Could not load submission status for {config_name}: {e}")
            return {exp_id: False for exp_id in dataset_experiments}
    
    def _check_completion_status(self, config_name: str, experiments_df) -> Dict[str, bool]:
        """Check completion status by examining filesystem for experiment_data.csv files."""
        # Find the engine params for this config - we need this to construct paths
        # For now, we'll need to pass this in or find another way to get it
        # This is a limitation of the current design that we'll need to address
        completion_status = {}
        
        try:
            # TODO: utilize the `list[ExperimentData]` instead of relying on the local dataset
            for data in experiments_iter(experiments_df):
                # We need engine_params to construct the journey_dir path
                # For now, mark as not completed if we can't determine the path
                # TODO: Improve this by passing engine_params or refactoring path construction
                completion_status[data.experiment_id] = False
                
        except Exception as e:
            _print(f"[bold yellow]Warning: Could not check completion status for {config_name}: {e}")
            
        return completion_status
    
    def _get_experiment_batch_id(self, config_name: str, experiment_id: str) -> Optional[str]:
        """Get batch_id for a specific experiment."""
        if not self.mapping_file.exists():
            return None
        
        try:
            with open(self.mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            if config_name in mapping_data:
                experiments = mapping_data[config_name].get("experiments", {})
                if experiment_id in experiments:
                    return experiments[experiment_id].get("batch_id")
                    
        except Exception:
            pass
        
        return None
    
    async def _update_experiment_batch_mapping(self, config_name: str, experiment_ids: List[str], 
                                             batch_ids: List[str]) -> None:
        """Update the detailed experiment-to-batch mapping file."""
        if not experiment_ids or not batch_ids:
            return
        
        # Load existing mapping
        mapping_data = {}
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r') as f:
                    mapping_data = json.load(f)
            except json.JSONDecodeError:
                _print(f"[bold yellow]Warning: Could not parse existing mapping file")
        
        # Initialize provider entry if needed
        if config_name not in mapping_data:
            mapping_data[config_name] = {
                "batches": {},
                "experiments": {}
            }
        
        # Record batch information
        for batch_id in batch_ids:
            if batch_id not in mapping_data[config_name]["batches"]:
                mapping_data[config_name]["batches"][batch_id] = {
                    "experiment_ids": [],
                    "submitted_at": asyncio.get_event_loop().time(),
                    "status": "submitted"
                }
            
            # Add experiments to this batch (avoid duplicates)
            existing_experiments = set(mapping_data[config_name]["batches"][batch_id]["experiment_ids"])
            new_experiments = [exp_id for exp_id in experiment_ids if exp_id not in existing_experiments]
            mapping_data[config_name]["batches"][batch_id]["experiment_ids"].extend(new_experiments)
        
        # Record reverse lookup: experiment -> batch
        for experiment_id in experiment_ids:
            mapping_data[config_name]["experiments"][experiment_id] = {
                "batch_id": batch_ids[0],  # Use first batch ID
                "submitted_at": asyncio.get_event_loop().time()
            }
        
        # Write updated mapping
        with open(self.mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
    
    async def _update_submitted_experiments_tracking(self, config_name: str, experiment_ids: List[str]) -> None:
        """Update simple submitted experiments tracking file."""
        submitted_data = {}
        
        # Load existing data
        if self.submitted_experiments_file.exists():
            try:
                with open(self.submitted_experiments_file, 'r') as f:
                    submitted_data = json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Update data for this provider
        if config_name not in submitted_data:
            submitted_data[config_name] = []
        
        # Add new experiment IDs (avoid duplicates)
        existing_ids = set(submitted_data[config_name])
        new_ids = [exp_id for exp_id in experiment_ids if exp_id not in existing_ids]
        submitted_data[config_name].extend(new_ids)
        
        # Write updated data
        with open(self.submitted_experiments_file, 'w') as f:
            json.dump(submitted_data, f, indent=2)