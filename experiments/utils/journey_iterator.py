from pathlib import Path
from typing import Iterator


def iterate_journey_directories(results_path: Path) -> Iterator[Path]:
    """
    Iterates through all journey directories for a specified dataset results path.
    
    Yields Path objects for directories matching the pattern:
    experiment_logs/{dataset_name}/{model_name}/{query}/{experiment_id}/
    
    Args:
        results_path: Path to the results directory (e.g., "experiment_logs/rating_sanity_check_dataset")
        
    Yields:
        Path: Directory paths for individual experiment journeys
    """
    if not results_path.exists() or not results_path.is_dir():
        raise ValueError(f"Results path does not exist or is not a directory: {results_path}")
            
    # Iterate through model directories
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        # Iterate through query directories
        for query_dir in model_dir.iterdir():
            if not query_dir.is_dir():
                continue
                
            # Iterate through experiment_id directories
            for experiment_dir in query_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue
                    
                yield experiment_dir