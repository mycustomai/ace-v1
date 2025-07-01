import os
from pathlib import Path
from typing import Optional, Union

from agent.src.environment import BaseShoppingEnvironment


class FilesystemShoppingEnvironment(BaseShoppingEnvironment):
    """
    A shopping environment that reads screenshots from the filesystem instead of capturing them.
    
    This is useful for replaying experiments or testing with pre-captured screenshots.
    """
    
    def __init__(self, screenshots_dir: Path, query: str, experiment_label: str, experiment_number: int, remote: bool = False):
        """
        Initialize the filesystem environment.
        
        Args:
            screenshots_dir: Base directory containing screenshots
            query: The product query (e.g., "mousepad")
            experiment_label: The experiment label (e.g., "control")
            experiment_number: The experiment number
            remote: If True, return GCS URLs instead of bytes
        """
        self.screenshots_dir = screenshots_dir
        self.query = query
        self.experiment_label = experiment_label
        self.experiment_number = experiment_number
        self.remote = remote
        
        # Build the expected screenshot path
        self.screenshot_path = (
            screenshots_dir / query / experiment_label / 
            f"{query}_{experiment_label}_{experiment_number}.png"
        )
    
    def capture_screenshot(self) -> Union[bytes, str]:
        """
        Read a screenshot from the filesystem or return a GCS URL.
        
        Returns:
            Union[bytes, str]: Screenshot data as bytes if remote=False, 
                              or public GCS URL as string if remote=True
            
        Raises:
            FileNotFoundError: If the screenshot file doesn't exist (local mode)
            ValueError: If GCS_BUCKET_NAME environment variable is not set (remote mode)
        """
        if self.remote:
            bucket_name = os.getenv('GCS_BUCKET_NAME')
            if not bucket_name:
                raise ValueError("GCS_BUCKET_NAME environment variable must be set for remote mode")
            
            # Find the screenshots directory in the path and construct the GCS path
            path_parts = self.screenshot_path.parts
            screenshots_index = path_parts.index('screenshots') if 'screenshots' in path_parts else -1
            
            if screenshots_index == -1:
                # If no 'screenshots' directory found, use the relative path from screenshots_dir
                relative_path = self.screenshot_path.relative_to(self.screenshots_dir)
                gcs_path = f"screenshots/{relative_path}"
            else:
                # Use the path starting from 'screenshots'
                gcs_path = '/'.join(path_parts[screenshots_index:])
            
            return f"https://storage.googleapis.com/{bucket_name}/{gcs_path}"
        else:
            if not self.screenshot_path.exists():
                raise FileNotFoundError(f"Screenshot not found: {self.screenshot_path}")
            
            return self.screenshot_path.read_bytes()