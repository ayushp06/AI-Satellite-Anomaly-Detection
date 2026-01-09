import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, List
import json

logger = logging.getLogger(__name__)

class DatasetV1Loader:
    """Loads and aggregates multiple dataset JSON files from dataset v1."""
    
    def __init__(self, dataset_dir: Path):
        """
        Args:
            dataset_dir: Directory containing the 100 dataset JSON files
        """
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
    
    def load_all_datasets(self) -> pd.DataFrame:
        """
        Load all JSON files from the dataset directory and concatenate.
        
        Returns:
            Combined DataFrame with all data
        """
        json_files = sorted(self.dataset_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.dataset_dir}")
        
        logger.info(f"Found {len(json_files)} dataset files")
        
        dfs = []
        for json_file in json_files:
            try:
                df = self._load_json_file(json_file)
                dfs.append(df)
                logger.debug(f"Loaded {json_file.name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {json_file.name}: {e}")
        
        if not dfs:
            raise ValueError("No datasets loaded successfully")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} total rows from {len(dfs)} files")
        
        return combined_df
    
    def _load_json_file(self, json_file: Path) -> pd.DataFrame:
        """
        Load a single JSON file and convert to DataFrame.
        Handles different JSON structures.
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Array of records
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check if it has a 'data' key with records
            if 'data' in data and isinstance(data['data'], list):
                df = pd.DataFrame(data['data'])
            # Check if keys are column names
            elif all(isinstance(v, list) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                # Single record dict
                df = pd.DataFrame([data])
        else:
            raise ValueError(f"Unexpected JSON structure in {json_file.name}")
        
        return df
    
    def get_dataset_count(self) -> int:
        """Get number of dataset files."""
        return len(list(self.dataset_dir.glob("*.json")))