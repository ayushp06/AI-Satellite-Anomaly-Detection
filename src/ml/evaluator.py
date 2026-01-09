import json
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config.phase4_config import Phase4Config
from src.ml.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

class Evaluator:
    """Handles model evaluation and visualization."""
    
    def __init__(self, config: Phase4Config):
        self.config = config
        self.plots_dir = config.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self):
        """
        Full evaluation pipeline.
        """
        logger.info("Loading model and data...")
        
        # Load model
        model_path = self.config.output_dir / "best_model.keras"
        model = tf.keras.models.load_model(str(model_path))
        
        # Load data
        pipeline = DataPipeline(self.config)
        X_train, y_train, X_val, y_val, X_test, y_test, _, _, test_ds = pipeline.load_and_process()
        
        # Get predictions
        logger.info("Generating predictions...")
        y_pred_proba = model.predict(X_test)
        
        if pipeline.num_classes == 2:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Confusion matrix
        logger.info("Creating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, pipeline.num_classes)
        
        # Classification report
        logger.info("Classification report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))
        
        # Save report
        report_path = self.config.output_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Plot sample predictions
        logger.info("Plotting sample predictions...")
        self._plot_sample_windows(X_test, y_test, y_pred, pipeline.num_classes)
        
        logger.info(f"Plots saved to {self.plots_dir}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, num_classes: int):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=range(num_classes),
                    yticklabels=range(num_classes))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        
        cm_path = self.plots_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")
    
    def _plot_sample_windows(self, X_test: np.ndarray, y_test: np.ndarray, 
                            y_pred: np.ndarray, num_classes: int, num_samples: int = 6):
        """Plot sample windows with predictions vs ground truth."""
        num_samples = min(num_samples, len(X_test))
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for plot_idx, window_idx in enumerate(indices):
            window = X_test[window_idx]
            true_label = y_test[window_idx]
            pred_label = y_pred[window_idx]
            
            ax = axes[plot_idx]
            timesteps = np.arange(len(window))
            
            # Plot each feature as a line
            for feature_idx in range(window.shape[1]):
                ax.plot(timesteps, window[:, feature_idx], 
                       label=self.config.feature_columns[feature_idx], alpha=0.7)
            
            # Add background color based on prediction correctness
            is_correct = true_label == pred_label
            color = "lightgreen" if is_correct else "lightcoral"
            ax.set_facecolor(color)
            
            title = f"Window {window_idx}: True={true_label}, Pred={pred_label}"
            if not is_correct:
                title += " [incorrect]"
            ax.set_title(title)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Normalized Value")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        sample_path = self.plots_dir / "sample_predictions.png"
        plt.savefig(sample_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Sample predictions saved to {sample_path}")
    
    def plot_training_history(self):
        """Plot training history."""
        history_path = self.config.output_dir / "training_history.json"
        if not history_path.exists():
            logger.warning(f"Training history not found: {history_path}")
            return
        
        with open(history_path) as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history["loss"], label="Train Loss")
        axes[0].plot(history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss vs Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history["accuracy"], label="Train Accuracy")
        axes[1].plot(history["val_accuracy"], label="Val Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy vs Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        history_path_plot = self.plots_dir / "training_history.png"
        plt.savefig(history_path_plot, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Training history plot saved to {history_path_plot}")
