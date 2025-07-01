import atexit
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from agent.src.environment import ShoppingEnvironment
from agent.src.types import TargetSite
from experiments.server import start_fastapi_server, stop_fastapi_server
from experiments.data_loader import experiments_iter, load_experiment_data
from sandbox import set_experiment_data


def is_valid_png(file_path: Path) -> bool:
    """
    Check if a file is a valid PNG image.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if the file is a valid PNG, False otherwise
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    
    try:
        # PNG files start with the magic bytes: 89 50 4E 47 0D 0A 1A 0A
        with open(file_path, 'rb') as f:
            header = f.read(8)
            return header == b'\x89PNG\r\n\x1a\n'
    except (OSError, IOError):
        return False


def get_missing_screenshots(combined_df: pd.DataFrame, dataset_path: str) -> List[Tuple[str, str, int]]:
    """
    Check which screenshots are missing for a given dataset.
    
    Args:
        combined_df: DataFrame containing experiment data
        dataset_path: Path to the dataset file
        
    Returns:
        List of tuples (query, experiment_label, experiment_number) for missing screenshots
    """
    # Extract dataset name from path
    dataset_name = Path(dataset_path).stem.replace('_dataset', '')
    
    # Base screenshots directory alongside the dataset
    dataset_dir = Path(dataset_path).parent
    screenshots_dir = dataset_dir / "screenshots" / dataset_name
    
    missing_screenshots = []
    
    for data in experiments_iter(combined_df):
        # Build expected screenshot path
        screenshot_dir = screenshots_dir / data.query / data.experiment_label
        filename = f"{data.query}_{data.experiment_label}_{data.experiment_number}.png"
        screenshot_path = screenshot_dir / filename
        
        if not is_valid_png(screenshot_path):
            missing_screenshots.append((data.query, data.experiment_label, data.experiment_number))
    
    return missing_screenshots


def ensure_all_screenshots_exist(combined_df: pd.DataFrame, dataset_path: str) -> bool:
    """
    Check if all required screenshots exist for a dataset.
    
    Args:
        combined_df: DataFrame containing experiment data
        dataset_path: Path to the dataset file
        
    Returns:
        bool: True if all screenshots exist and are valid, False otherwise
    """
    missing = get_missing_screenshots(combined_df, dataset_path)
    
    if missing:
        print(f"Missing {len(missing)} screenshots:")
        for query, experiment_label, experiment_number in missing:
            print(f"  - {query}_{experiment_label}_{experiment_number}.png")
        return False
    
    print("All required screenshots exist and are valid.")
    return True


def collect_screenshots(combined_df: pd.DataFrame, dataset_path: str):
    """
    Runs through the experiment df, starts the webserver, and collects screenshots for each experiment.
    
    Saves screenshots to filesystem hierarchy alongside the dataset:
    
    screenshots/{dataset_name}/{query}/{experiment_label}/
    └── {query}_{experiment_label}_{experiment_number}.png
    
    Example structure:
    datasets/sanity-checks/screenshots/price_sanity_check/
    ├── mousepad/
    │   ├── control/
    │   │   ├── mousepad_control_1.png
    │   │   └── mousepad_control_2.png
    │   └── experimental/
    │       ├── mousepad_experimental_1.png
    │       └── mousepad_experimental_2.png
    └── toothpaste/
        ├── control/
        │   ├── toothpaste_control_1.png
        │   └── toothpaste_control_2.png
        └── experimental/
            ├── toothpaste_experimental_1.png
            └── toothpaste_experimental_2.png
    """
    server_thread = start_fastapi_server()
    atexit.register(stop_fastapi_server)

    # Extract dataset name from path
    dataset_name = Path(dataset_path).stem.replace('_dataset', '')
    
    # Base screenshots directory alongside the dataset
    dataset_dir = Path(dataset_path).parent
    screenshots_dir = dataset_dir / "screenshots" / dataset_name

    # manually start
    env = ShoppingEnvironment(TargetSite.MOCKAMAZON)
    env._init_driver()

    for data in experiments_iter(combined_df):
        # Create directory structure: screenshots/{dataset_name}/{query}/{experiment_label}/
        screenshot_dir = screenshots_dir / data.query / data.experiment_label
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename: {query}_{experiment_label}_{experiment_number}.png
        filename = f"{data.query}_{data.experiment_label}_{data.experiment_number}.png"
        screenshot_path = screenshot_dir / filename
        
        # Skip if screenshot already exists and is valid
        if is_valid_png(screenshot_path):
            print(f"Skipping existing screenshot: {screenshot_path}")
            continue
        
        set_experiment_data(data.experiment_df)
        env._navigate_to_product_search(data.query)
        screenshot = env.capture_screenshot()
        
        # Save screenshot to file
        with open(screenshot_path, 'wb') as f:
            f.write(screenshot)
            
        print(f"Saved screenshot: {screenshot_path}")


if __name__ == "__main__":
    dataset_path = "datasets/sanity-checks/price_sanity_check_dataset.csv"
    combined_df = load_experiment_data(dataset_path)

    collect_screenshots(combined_df, dataset_path)