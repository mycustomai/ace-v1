"""
Screenshot validation service for batch and screenshot runtimes.

Centralizes all screenshot validation and collection operations.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd

from rich import print as _print

from experiments.utils.screenshot_collector import ensure_all_screenshots_exist, collect_screenshots


class ScreenshotValidationService:
    """Unified screenshot validation across all runtimes."""
    
    def __init__(self, screenshots_dir: Path):
        self.screenshots_dir = Path(screenshots_dir)
    
    def validate_all_screenshots(self, experiments_df: pd.DataFrame, dataset_path: Optional[str] = None) -> bool:
        """
        Validate that all required screenshots exist.
        
        Args:
            experiments_df: DataFrame containing experiments
            dataset_path: Optional path to dataset for regeneration
            
        Returns:
            True if all screenshots exist and are valid
        """
        _print("[bold blue]Validating screenshot availability...")
        
        if dataset_path and not ensure_all_screenshots_exist(experiments_df, dataset_path):
            _print("[bold yellow]Some screenshots are missing. Attempting to regenerate...")
            
            # Try to regenerate missing screenshots
            success = self.regenerate_screenshots(experiments_df, dataset_path)
            if not success:
                return False
        elif not dataset_path:
            # Just check existence without regeneration capability
            missing_screenshots = self.find_missing_screenshots(experiments_df)
            if missing_screenshots:
                _print(f"[bold red]Found {len(missing_screenshots)} missing screenshots")
                _print("[bold red]Cannot regenerate without dataset_path")
                return False
        
        _print("[bold green]All required screenshots are available and valid.")
        return True
    
    def regenerate_screenshots(self, experiments_df: pd.DataFrame, dataset_path: str) -> bool:
        """
        Regenerate missing screenshots.
        
        Args:
            experiments_df: DataFrame containing experiments
            dataset_path: Path to dataset file
            
        Returns:
            True if regeneration was successful
        """
        try:
            _print("[bold blue]Regenerating missing screenshots...")
            collect_screenshots(experiments_df, dataset_path)
            
            # Validate again after regeneration
            if ensure_all_screenshots_exist(experiments_df, dataset_path):
                _print("[bold green]All screenshots successfully regenerated.")
                return True
            else:
                _print("[bold red]Screenshot regeneration failed. Some screenshots are still missing.")
                return False
                
        except Exception as e:
            _print(f"[bold red]Error during screenshot regeneration: {e}")
            return False
    
    def find_missing_screenshots(self, experiments_df: pd.DataFrame) -> List[Path]:
        """
        Find screenshots that are missing or invalid.
        
        Args:
            experiments_df: DataFrame containing experiments
            
        Returns:
            List of missing screenshot paths
        """
        missing_screenshots = []
        
        try:
            from experiments.data_loader import experiments_iter
            
            for data in experiments_iter(experiments_df):
                screenshot_path = self._get_screenshot_path(data.query, data.experiment_label, data.experiment_number)
                
                if not screenshot_path.exists() or not self._is_valid_screenshot(screenshot_path):
                    missing_screenshots.append(screenshot_path)
                    
        except Exception as e:
            _print(f"[bold yellow]Warning: Error checking screenshot existence: {e}")
        
        return missing_screenshots
    
    def get_screenshot_paths_for_experiments(self, experiments_df: pd.DataFrame) -> List[Path]:
        """
        Get all screenshot paths for the given experiments.
        
        Args:
            experiments_df: DataFrame containing experiments
            
        Returns:
            List of screenshot paths
        """
        screenshot_paths = []
        
        try:
            from experiments.data_loader import experiments_iter
            
            for data in experiments_iter(experiments_df):
                screenshot_path = self._get_screenshot_path(data.query, data.experiment_label, data.experiment_number)
                screenshot_paths.append(screenshot_path)
                
        except Exception as e:
            _print(f"[bold yellow]Warning: Error getting screenshot paths: {e}")
        
        return screenshot_paths
    
    def _get_screenshot_path(self, query: str, experiment_label: str, experiment_number: int) -> Path:
        """
        Get the expected path for a screenshot.
        
        Args:
            query: Search query
            experiment_label: Experiment label
            experiment_number: Experiment number
            
        Returns:
            Expected screenshot path
        """
        return (
            self.screenshots_dir / query / experiment_label / 
            f"{query}_{experiment_label}_{experiment_number}.png"
        )
    
    def _is_valid_screenshot(self, screenshot_path: Path) -> bool:
        """
        Check if a screenshot file is valid.
        
        Args:
            screenshot_path: Path to screenshot file
            
        Returns:
            True if screenshot is valid
        """
        try:
            if not screenshot_path.exists():
                return False
            
            # Check file size (empty files are invalid)
            if screenshot_path.stat().st_size == 0:
                return False
            
            # TODO: Add more validation if needed (e.g., image format validation)
            return True
            
        except Exception:
            return False
    
    def ensure_screenshots_directory_exists(self) -> None:
        """Ensure the screenshots directory exists."""
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    def get_screenshot_stats(self, experiments_df: pd.DataFrame) -> dict:
        """
        Get statistics about screenshot availability.
        
        Args:
            experiments_df: DataFrame containing experiments
            
        Returns:
            Dictionary with screenshot statistics
        """
        total_screenshots = 0
        existing_screenshots = 0
        missing_screenshots = []
        
        try:
            from experiments.data_loader import experiments_iter
            
            for data in experiments_iter(experiments_df):
                total_screenshots += 1
                screenshot_path = self._get_screenshot_path(data.query, data.experiment_label, data.experiment_number)
                
                if self._is_valid_screenshot(screenshot_path):
                    existing_screenshots += 1
                else:
                    missing_screenshots.append(screenshot_path)
                    
        except Exception as e:
            _print(f"[bold yellow]Warning: Error calculating screenshot stats: {e}")
        
        return {
            'total': total_screenshots,
            'existing': existing_screenshots,
            'missing': len(missing_screenshots),
            'missing_paths': missing_screenshots,
            'completion_rate': existing_screenshots / total_screenshots if total_screenshots > 0 else 0
        }