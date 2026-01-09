from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class Phase4Config:
    """Configuration for Phase 4 ML pipeline."""
    
    # Data paths - can be single CSV or directory of CSVs (dataset v1)
    data_path: Path = Path("data/dataset_v1")  # Changed default to dataset_v1 folder
    output_dir: Path = Path("results/phase4")
    
    # Data source type
    data_source: str = "dataset_v1"  # "dataset_v1" (multiple CSVs) or "single_csv"
    
    # Time-series window parameters
    window_seconds: float = 5.0
    sample_rate_hz: int = 10
    stride: int = 1
    
    # Features and labels
    feature_columns: List[str] = field(default_factory=lambda: [
        "q0", "q1", "q2", "q3", "wx", "wy", "wz"
    ])
    label_column: str = "label"
    
    # Data split ratios (chronological split)
    test_split: float = 0.15
    val_split: float = 0.15
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    random_seed: int = 42
    
    # Model architecture
    model_type: str = "lstm"  # "lstm" or "cnn1d"
    lstm_units: int = 64
    cnn1d_filters: int = 32
    dropout_rate: float = 0.3
    
    # Training callbacks
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    
    # Preprocessing options
    normalize_quaternions: bool = True
    
    @property
    def window_size(self) -> int:
        """Calculate window size in samples."""
        return int(self.window_seconds * self.sample_rate_hz)
    
    @property
    def num_features(self) -> int:
        """Number of input features."""
        return len(self.feature_columns)
    
    def __post_init__(self):
        """Validate configuration."""
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        
        if self.model_type not in ("lstm", "cnn1d"):
            raise ValueError(f"model_type must be 'lstm' or 'cnn1d', got {self.model_type}")
        
        if self.data_source not in ("dataset_v1", "single_csv"):
            raise ValueError(f"data_source must be 'dataset_v1' or 'single_csv', got {self.data_source}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)