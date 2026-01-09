import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
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
        Load data, create windows, normalize, and split.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, 
                     train_ds, val_ds, test_ds)
        """
        # Load and validate data
        df = self._load_data()
        df = self._validate_and_prepare(df)
        
        logger.info(f"Loaded dataset: {len(df)} total rows")
        
        # Normalize quaternions if configured
        if self.config.normalize_quaternions:
            df = self._normalize_quaternions(df)
        
        # Create windows
        X, y = self._create_windows(df)
        
        if len(X) == 0:
            raise ValueError("No windows created. Check window_size and data length.")
        
        # Determine number of classes
        self.num_classes = int(np.max(y)) + 1
        logger.info(f"Detected {self.num_classes} classes")
        logger.info(f"Label distribution: {np.bincount(y)}")
        
        # Chronological train/val/test split
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_data(X, y)
        
        # Fit scaler on train set, transform all
        X_train_norm = self._normalize_features(X_train, X_val, X_test)
        X_val_norm = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_norm = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Create tf.data.Dataset objects
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
        label_data = df[self.config.label_column].values
        
        num_samples = len(feature_data)
        window_size = self.config.window_size
        stride = self.config.stride
        
        for start in range(0, num_samples - window_size + 1, stride):
            end = start + window_size
            window_features = feature_data[start:end]
            window_labels = label_data[start:end]
            
            # Label assignment: most frequent (mode) for multi-class, max for binary
            unique_labels = np.unique(window_labels)
            if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
                # Binary: use max (if any fault, label is 1)
                window_label = int(np.max(window_labels))
            else:
                # Multi-class: use mode
                window_label = int(np.bincount(window_labels.astype(int)).argmax())
            
            X_list.append(window_features)
            y_list.append(window_label)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        logger.info(f"Created {len(X)} windows of size {window_size}")
        return X, y
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Chronological train/val/test split to avoid leakage."""
        n = len(X)
        n_test = int(n * self.config.test_split)
        n_val = int(n * self.config.val_split)
        n_train = n - n_test - n_val
        
        # Chronological order
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
        
        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    
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