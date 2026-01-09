import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import json

logger = logging.getLogger(__name__)

class DatasetV1Loader:
    """Loads and aggregates telemetry from dataset v1."""
    
    def __init__(self, dataset_dir: Path):
        """
        Args:
            dataset_dir: Directory containing dataset_summary.json and metadata_run_*.json files
        """
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
    
    def load_all_datasets(self) -> pd.DataFrame:
        """
        Load all run JSON files (metadata_run_*.json) and convert to telemetry DataFrame.
        
        Returns:
            Combined DataFrame with all telemetry data and labels
        """
        telemetry_files = sorted(self.dataset_dir.glob("telemetry_run_*.parquet"))
        metadata_files = sorted(self.dataset_dir.glob("metadata_run_*.json"))

        if telemetry_files:
            logger.info(f"Found {len(telemetry_files)} telemetry files")
        elif metadata_files:
            logger.info(f"Found {len(metadata_files)} metadata files (legacy inline telemetry)")
        else:
            raise FileNotFoundError(
                f"No telemetry_run_*.parquet or metadata_run_*.json files found in {self.dataset_dir}"
            )
        
        dfs = []
        if telemetry_files:
            metadata_lookup = {self._extract_run_id(f): f for f in metadata_files}
            for telemetry_file in telemetry_files:
                run_id = self._extract_run_id(telemetry_file)
                metadata_file = metadata_lookup.get(run_id)
                try:
                    df = self._load_telemetry_file(telemetry_file, metadata_file, run_id)
                    if df is not None and len(df) > 0:
                        dfs.append(df)
                        logger.debug(f"Loaded {telemetry_file.name}: {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Failed to load {telemetry_file.name}: {e}")
        else:
            for run_file in metadata_files:
                try:
                    df = self._load_run_file(run_file)
                    if df is not None and len(df) > 0:
                        dfs.append(df)
                        logger.debug(f"Loaded {run_file.name}: {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Failed to load {run_file.name}: {e}")
        
        if not dfs:
            raise ValueError("No run data loaded successfully")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined telemetry: {len(combined_df)} total rows from {len(dfs)} runs")
        
        return combined_df
    
    def _load_telemetry_file(
        self,
        telemetry_file: Path,
        metadata_file: Optional[Path],
        run_id: int
    ) -> pd.DataFrame:
        """
        Load a telemetry parquet file and ensure labels are present.
        """
        df = pd.read_parquet(telemetry_file)
        df = self._standardize_telemetry_columns(df)
        df["run_id"] = run_id

        if "fault" not in df.columns:
            metadata = self._load_metadata(metadata_file)
            df["fault"] = self._labels_from_metadata(df, metadata)
        else:
            df["fault"] = df["fault"].fillna(0).astype(int)

        return df

    def _load_run_file(self, run_file: Path) -> pd.DataFrame:
        """
        Load a single run JSON file and convert to DataFrame.
        Extracts telemetry data and adds label based on fault_windows.
        """
        with open(run_file, 'r') as f:
            run_data = json.load(f)
        
        # Extract metadata
        run_id = run_data.get('run_id', 0)
        fault_injected = run_data.get('fault_injected', False)
        fault_windows = run_data.get('fault_windows', [])
        dt = run_data.get('dt', 0.1)
        
        # Extract telemetry - could be in different locations
        telemetry = None
        if 'telemetry' in run_data:
            telemetry = run_data['telemetry']
        elif 'data' in run_data:
            telemetry = run_data['data']
        
        if telemetry is None:
            logger.warning(f"No telemetry data found in {run_file.name}")
            return None
        
        # Convert telemetry list to DataFrame
        if isinstance(telemetry, list):
            df = pd.DataFrame(telemetry)
            df = self._standardize_telemetry_columns(df)
        else:
            logger.warning(f"Unexpected telemetry type in {run_file.name}: {type(telemetry)}")
            return None
        
        # Add run_id
        df['run_id'] = run_id
        
        if "fault" not in df.columns:
            df["fault"] = self._labels_from_metadata(
                df,
                {
                    "fault_injected": fault_injected,
                    "fault_windows": fault_windows,
                }
            )
        else:
            df["fault"] = df["fault"].fillna(0).astype(int)
        
        return df

    @staticmethod
    def _extract_run_id(path: Path) -> int:
        """Extract run id from a telemetry/metadata filename."""
        stem = path.stem
        run_id_str = stem.split("_")[-1]
        try:
            return int(run_id_str)
        except ValueError:
            return 0

    @staticmethod
    def _load_metadata(metadata_file: Optional[Path]) -> Optional[dict]:
        if metadata_file is None or not metadata_file.exists():
            return None
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _standardize_telemetry_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Expand nested q/w columns into q0..q3 and w0..w2 if needed."""
        if "q" in df.columns:
            q_expanded = pd.DataFrame(df["q"].tolist(), columns=["q0", "q1", "q2", "q3"])
            df = df.drop(columns=["q"]).join(q_expanded)
        if "w" in df.columns:
            w_expanded = pd.DataFrame(df["w"].tolist(), columns=["w0", "w1", "w2"])
            df = df.drop(columns=["w"]).join(w_expanded)
        return df

    @staticmethod
    def _labels_from_metadata(df: pd.DataFrame, metadata: Optional[dict]) -> np.ndarray:
        """Generate binary fault labels from metadata fault windows."""
        labels = np.zeros(len(df), dtype=np.int32)
        if not metadata:
            return labels

        if not metadata.get("fault_injected") or not metadata.get("fault_windows"):
            return labels

        for fault_window in metadata.get("fault_windows", []):
            if isinstance(fault_window, dict):
                if "start_index" in fault_window and "end_index" in fault_window:
                    start_idx = int(fault_window["start_index"])
                    end_idx = int(fault_window["end_index"])
                    labels[start_idx:end_idx + 1] = 1
                elif "start" in fault_window and "end" in fault_window and "t" in df.columns:
                    start_t = float(fault_window["start"])
                    end_t = float(fault_window["end"])
                    mask = (df["t"] >= start_t) & (df["t"] <= end_t)
                    labels[mask.to_numpy()] = 1
            elif isinstance(fault_window, (list, tuple)) and len(fault_window) >= 2:
                start_idx, end_idx = int(fault_window[0]), int(fault_window[1])
                labels[start_idx:end_idx + 1] = 1
        return labels
    
    def get_dataset_count(self) -> int:
        """Get number of run files."""
        telemetry_files = list(self.dataset_dir.glob("telemetry_run_*.parquet"))
        if telemetry_files:
            return len(telemetry_files)
        return len(list(self.dataset_dir.glob("metadata_run_*.json")))
