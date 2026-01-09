import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config.phase4_config import Phase4Config
from src.ml.data_pipeline import DataPipeline
from src.ml.model_builder import build_model

logger = logging.getLogger(__name__)

class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: Phase4Config):
        self.config = config
        self.model = None
        self.history = None
    
    def train(self) -> dict:
        """
        Main training pipeline.
        
        Returns:
            Dictionary of metrics
        """
        logger.info("=" * 60)
        logger.info("Phase 4: Fault Detection ML Training")
        logger.info("=" * 60)
        
        # Load and process data
        logger.info("Loading data...")
        pipeline = DataPipeline(self.config)
        X_train, y_train, X_val, y_val, X_test, y_test, train_ds, val_ds, test_ds = pipeline.load_and_process()
        
        # Build model
        logger.info(f"Building {self.config.model_type} model...")
        self.model = build_model(self.config, pipeline.num_classes)
        self.model.summary()
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Train
        logger.info("Training model...")
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loss, *test_metrics = self.model.evaluate(test_ds, verbose=0)
        
        metrics = {
            "test_loss": float(test_loss),
            "num_classes": pipeline.num_classes,
            "model_type": self.config.model_type,
            "window_size": self.config.window_size,
            "batch_size": self.config.batch_size,
            "epochs_trained": len(self.history.history["loss"])
        }
        
        if pipeline.num_classes == 2:
            metrics["test_accuracy"] = float(test_metrics[0])
            metrics["test_precision"] = float(test_metrics[1])
            metrics["test_recall"] = float(test_metrics[2])
        else:
            metrics["test_accuracy"] = float(test_metrics[0])
        
        # Save artifacts
        logger.info("Saving artifacts...")
        self._save_artifacts(metrics, pipeline, X_test, y_test, test_ds)
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        
        return metrics
    
    def _setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.config.output_dir / "best_model.keras"
        callbacks.append(keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ))
        
        # Early stopping
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))
        
        # Reduce LR on plateau
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ))
        
        return callbacks
    
    def _save_artifacts(self, metrics: dict, pipeline: DataPipeline, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       test_ds: tf.data.Dataset):
        """Save model, scaler, config, and metrics."""
        # Save model
        model_path = self.config.output_dir / "best_model.keras"
        self.model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        scaler_path = self.config.output_dir / "scaler.pkl"
        pipeline.save_scaler(scaler_path)
        
        # Save config
        config_path = self.config.output_dir / "config.json"
        with open(config_path, "w") as f:
            config_dict = {
                "window_size": self.config.window_size,
                "window_seconds": self.config.window_seconds,
                "sample_rate_hz": self.config.sample_rate_hz,
                "num_features": self.config.num_features,
                "model_type": self.config.model_type,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            }
            json.dump(config_dict, f, indent=2)
        logger.info(f"Config saved to {config_path}")
        
        # Save metrics
        metrics_path = self.config.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save training history
        history_path = self.config.output_dir / "training_history.json"
        history_dict = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)
        logger.info(f"Training history saved to {history_path}")