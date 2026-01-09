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
        Load all JSON files from the dataset directory, extract telemetry paths,
        and concatenate all telemetry data.
        
        Returns:
            Combined DataFrame with all telemetry data
        """
        json_files = sorted(self.dataset_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.dataset_dir}")
        
        logger.info(f"Found {len(json_files)} dataset metadata files")
        
        dfs = []
        for json_file in json_files:
            try:
                df = self._load_telemetry_from_json(json_file)
                if df is not None and len(df) > 0:
                    dfs.append(df)
                    logger.debug(f"Loaded telemetry from {json_file.name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load telemetry from {json_file.name}: {e}")
        
        if not dfs:
            raise ValueError("No telemetry data loaded successfully")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined telemetry: {len(combined_df)} total rows from {len(dfs)} datasets")
        
        return combined_df
    
    def _load_telemetry_from_json(self, json_file: Path) -> pd.DataFrame:
        """
        Load a JSON metadata file, find the output directory,
        and load all telemetry CSV files from it.
        """
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract output directory path
        output_dir = metadata.get('output_directory')
        if not output_dir:
            logger.warning(f"No output_directory in {json_file.name}")
            return None
        
        output_path = Path(output_dir)
        if not output_path.exists():
            logger.warning(f"Output directory not found: {output_dir}")
            return None
        
        # Find all telemetry CSV files in the output directory
        telemetry_files = sorted(output_path.glob("*.csv"))
        if not telemetry_files:
            logger.warning(f"No CSV telemetry files in {output_dir}")
            return None
        
        # Load and combine all telemetry files from this dataset
        dfs = []
        for csv_file in telemetry_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {csv_file.name}: {e}")
        
        if not dfs:
            return None
        
        combined = pd.concat(dfs, ignore_index=True)
        return combined
    
    def get_dataset_count(self) -> int:
        """Get number of dataset metadata files."""
        return len(list(self.dataset_dir.glob("*.json")))