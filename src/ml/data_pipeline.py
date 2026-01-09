import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import logging

from src.ml.dataset_loader import DatasetV1Loader

logger = logging.getLogger(__name__)

class DataPipeline:
    """Handles data loading, windowing, normalization, and splitting."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.num_classes = None
    
    def load_and_process(self) -> Tuple[
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
    ]:
        """
        Load data, split, create windows, normalize, and build datasets.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test,
                     train_ds, val_ds, test_ds)
        """
        df = self._load_data()
        df = self._validate_and_prepare(df)

        logger.info(f"Loaded dataset: {len(df)} total rows")
        self._log_label_distribution(df[self.config.label_column].values, "raw rows")
        self._check_class_balance(df[self.config.label_column].values, "raw rows")

        if self.config.normalize_quaternions:
            df = self._normalize_quaternions(df)

        train_df, val_df, test_df = self._split_dataframe(df)
        self._log_split_distribution(train_df, val_df, test_df)

        X_train, y_train = self._create_windows(train_df)
        X_val, y_val = self._create_windows(val_df)
        X_test, y_test = self._create_windows(test_df)

        self._log_label_distribution(y_train, "train windows")
        self._log_label_distribution(y_val, "val windows")
        self._log_label_distribution(y_test, "test windows")
        self._check_class_balance(np.concatenate([y_train, y_val, y_test]), "all windows")
        self._validate_split_distribution(y_train, "train windows")
        self._validate_split_distribution(y_val, "val windows")
        self._validate_split_distribution(y_test, "test windows")

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError("One or more splits produced zero windows.")

        self.num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
        logger.info(f"Detected {self.num_classes} classes")

        X_train_norm = self._normalize_features(X_train, X_val, X_test)
        X_val_norm = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_norm = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        train_ds = self._create_dataset(X_train_norm, y_train, shuffle=True)
        val_ds = self._create_dataset(X_val_norm, y_val, shuffle=False)
        test_ds = self._create_dataset(X_test_norm, y_test, shuffle=False)

        logger.info(f"Split - Train: {len(X_train_norm)}, Val: {len(X_val_norm)}, Test: {len(X_test_norm)}")

        return X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test, train_ds, val_ds, test_ds

    def _load_data(self) -> pd.DataFrame:
        """Load data from single JSON, CSV, or multiple JSONs (dataset v1)."""
        try:
            if self.config.data_source == "dataset_v1":
                logger.info(f"Loading dataset v1 from {self.config.data_path}")
                loader = DatasetV1Loader(self.config.data_path)
                df = loader.load_all_datasets()
            else:  # single file (JSON or CSV)
                data_path = self.config.data_path
                logger.info(f"Loading data from {data_path}")
                
                if data_path.suffix.lower() == '.json':
                    import json
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.DataFrame([data])
                else:  # CSV
                    df = pd.read_csv(data_path)
            
            logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
    
    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns and sort by timestamp."""
        df = self._align_feature_columns(df)

        required_cols = self.config.feature_columns + [self.config.label_column]
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            logger.warning(f"Missing columns: {missing}")
            logger.info(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")

        # Handle timestamp column for sorting
        timestamp_col = None
        for col in ["timestamp", "t", "time"]:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col:
            df = df.sort_values(by=timestamp_col).reset_index(drop=True)
            logger.info(f"Sorted by {timestamp_col}")
        else:
            logger.warning("No timestamp column found; assuming data is pre-sorted")

        return df

    def _align_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create alias columns when alternate names are present."""
        alias_map = {
            "w0": "wx",
            "w1": "wy",
            "w2": "wz",
            "wx": "w0",
            "wy": "w1",
            "wz": "w2",
        }
        for expected in self.config.feature_columns:
            if expected not in df.columns:
                alias = alias_map.get(expected)
                if alias and alias in df.columns:
                    df[expected] = df[alias]

        if self.config.label_column not in df.columns:
            if self.config.label_column == "fault" and "label" in df.columns:
                df["fault"] = df["label"]
        return df

    def _log_label_distribution(self, labels: np.ndarray, context: str):
        labels = labels.astype(int)
        counts = np.bincount(labels, minlength=2)
        total = counts.sum()
        ratios = counts / total if total else counts
        logger.info(f"{context} label distribution: {counts} (ratio {ratios})")

    def _check_class_balance(self, labels: np.ndarray, context: str):
        labels = labels.astype(int)
        counts = np.bincount(labels, minlength=2)
        total = counts.sum()
        if total == 0:
            raise ValueError(f"No labels available for {context}")
        dominant_ratio = counts.max() / total
        if dominant_ratio > 0.95:
            raise ValueError(
                f"Class imbalance too high in {context}: {counts} (ratio {dominant_ratio:.2%})"
            )

    def _normalize_quaternions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize quaternion vectors to unit length."""
        q_cols = ["q0", "q1", "q2", "q3"]
        if not all(col in df.columns for col in q_cols):
            logger.warning("Quaternion columns not found; skipping normalization")
            return df
        
        q_data = df[q_cols].values
        q_norm = np.linalg.norm(q_data, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1  # Avoid division by zero
        
        df[q_cols] = q_data / q_norm
        logger.info("Quaternions normalized to unit length")
        return df
    
    def _create_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows from time-series data."""
        X_list = []
        y_list = []

        feature_data = df[self.config.feature_columns].values
        label_data = df[self.config.label_column].values.astype(int)

        num_samples = len(feature_data)
        window_size = self.config.window_size
        stride = self.config.stride

        if self.config.window_label_mode == "center":
            center_offset = window_size // 2
        else:
            center_offset = None

        for start_idx in range(0, num_samples - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_features = feature_data[start_idx:end_idx]
            window_labels = label_data[start_idx:end_idx]

            if self.config.window_label_mode == "center":
                window_label = int(window_labels[center_offset])
            elif self.config.window_label_mode == "percentage":
                fault_ratio = float(np.mean(window_labels))
                window_label = int(fault_ratio >= self.config.fault_ratio_threshold)
            else:
                raise ValueError(f"Unknown window_label_mode: {self.config.window_label_mode}")

            X_list.append(window_features)
            y_list.append(window_label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        logger.info(f"Created {len(X)} windows of size {window_size}")
        logger.info(f"Window labeling mode: {self.config.window_label_mode}")
        return X, y

    def _split_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Block-aware time split that preserves class presence."""
        labels = df[self.config.label_column].values.astype(int)
        blocks = self._build_blocks(labels)
        train_end, val_end = self._select_block_splits(blocks)

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)

        logger.info(
            f"Block-aware split indices: train_end={train_end}, val_end={val_end}, total={len(df)}"
        )
        logger.info("Chronological split applied (no shuffling).")
        return train_df, val_df, test_df

    def _build_blocks(self, labels: np.ndarray) -> List[Dict[str, int]]:
        """Build contiguous label blocks."""
        if len(labels) == 0:
            raise ValueError("No labels available to build blocks")

        blocks = []
        start = 0
        current = int(labels[0])
        for idx in range(1, len(labels)):
            value = int(labels[idx])
            if value != current:
                blocks.append({"start": start, "end": idx, "label": current})
                start = idx
                current = value
        blocks.append({"start": start, "end": len(labels), "label": current})
        return blocks

    def _select_block_splits(self, blocks: List[Dict[str, int]]) -> Tuple[int, int]:
        """Select block boundaries that keep faults in every split."""
        total = blocks[-1]["end"]
        target_train = int(total * (1.0 - self.config.val_split - self.config.test_split))
        target_val = int(total * self.config.val_split)

        boundaries = [block["end"] for block in blocks[:-1]]
        if not boundaries:
            raise ValueError("Not enough blocks to split data.")

        best = None
        for train_end in boundaries:
            for val_end in boundaries:
                if val_end <= train_end or val_end >= total:
                    continue

                splits = {
                    "train": (0, train_end),
                    "val": (train_end, val_end),
                    "test": (val_end, total),
                }
                if not self._splits_have_both_classes(blocks, splits):
                    continue
                if not self._splits_meet_fault_ratio(blocks, splits, min_ratio=0.01):
                    continue

                train_size = train_end
                val_size = val_end - train_end
                test_size = total - val_end
                score = abs(train_size - target_train) + abs(val_size - target_val) + abs(test_size - (total - target_train - target_val))
                if best is None or score < best[0]:
                    best = (score, train_end, val_end)

        if best is None:
            raise ValueError(
                "Could not find block-aware split that preserves class presence in all splits."
            )
        _, train_end, val_end = best
        return train_end, val_end

    def _splits_have_both_classes(self, blocks: List[Dict[str, int]], splits: Dict[str, Tuple[int, int]]) -> bool:
        for _, (start, end) in splits.items():
            counts = self._count_labels_in_range(blocks, start, end)
            if counts[0] == 0 or counts[1] == 0:
                return False
        return True

    def _splits_meet_fault_ratio(
        self, blocks: List[Dict[str, int]], splits: Dict[str, Tuple[int, int]], min_ratio: float
    ) -> bool:
        for _, (start, end) in splits.items():
            counts = self._count_labels_in_range(blocks, start, end)
            total = counts[0] + counts[1]
            if total == 0:
                return False
            ratio = counts[1] / total
            if ratio < min_ratio:
                return False
        return True

    def _count_labels_in_range(self, blocks: List[Dict[str, int]], start: int, end: int) -> np.ndarray:
        counts = np.zeros(2, dtype=int)
        for block in blocks:
            block_start = block["start"]
            block_end = block["end"]
            if block_end <= start or block_start >= end:
                continue
            overlap_start = max(start, block_start)
            overlap_end = min(end, block_end)
            overlap = overlap_end - overlap_start
            if overlap > 0:
                counts[block["label"]] += overlap
        return counts

    def _log_split_distribution(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        self._log_label_distribution(train_df[self.config.label_column].values, "train rows")
        self._log_label_distribution(val_df[self.config.label_column].values, "val rows")
        self._log_label_distribution(test_df[self.config.label_column].values, "test rows")

    def _validate_split_distribution(self, labels: np.ndarray, context: str):
        labels = labels.astype(int)
        counts = np.bincount(labels, minlength=2)
        total = counts.sum()
        if total == 0:
            raise ValueError(f"No samples available for {context}")
        fault_ratio = counts[1] / total
        if fault_ratio < 0.01:
            raise ValueError(
                f"{context} has insufficient fault windows: {counts} (ratio {fault_ratio:.2%})"
            )

    def _normalize_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                           X_test: np.ndarray) -> np.ndarray:
        """Fit scaler on train, transform all."""
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        X_train_norm = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
        logger.info("Features normalized (StandardScaler fit on train)")
        return X_train_norm
    
    def _create_dataset(self, X: np.ndarray, y: np.ndarray, 
                       shuffle: bool = False) -> tf.data.Dataset:
        """Create tf.data.Dataset with batching and optional shuffling."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(X), 1000), 
                                      seed=self.config.random_seed)
        
        dataset = dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def save_scaler(self, path: Path):
        """Save scaler for inference."""
        import joblib
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")
    
    @staticmethod
    def load_scaler(path: Path):
        """Load scaler from disk."""
        import joblib
        return joblib.load(path)