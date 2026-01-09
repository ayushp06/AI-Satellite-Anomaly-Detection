import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc

from config.phase4_config import Phase4Config
from src.ml.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles model evaluation and visualization."""

    def __init__(self, config: Phase4Config):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> Dict[str, float]:
        """Full evaluation pipeline."""
        logger.info("Loading model and data...")

        model_path = self.output_dir / "model.keras"
        model = tf.keras.models.load_model(str(model_path))

        pipeline = DataPipeline(self.config)
        X_train, y_train, X_val, y_val, X_test, y_test, _, _, _ = pipeline.load_and_process()

        logger.info("Generating predictions...")
        y_pred_proba = model.predict(X_test, verbose=0).reshape(-1)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        pred_counts = np.bincount(y_pred, minlength=2)
        pred_ratio = pred_counts / pred_counts.sum() if pred_counts.sum() else pred_counts
        logger.info(f"Prediction distribution: {pred_counts} (ratio {pred_ratio})")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        self._plot_confusion_matrices(cm)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        self._plot_roc_curve(fpr, tpr, roc_auc)

        self._plot_sample_windows_by_outcome(X_test, y_test, y_pred, y_pred_proba)

        report = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(roc_auc),
        }

        report_path = self.output_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation summary: {report}")
        return report

    def _plot_confusion_matrices(self, cm: np.ndarray):
        """Plot raw and normalized confusion matrices."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(cm, cmap="Blues")
        axes[0].set_title("Confusion Matrix (Counts)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0].text(j, i, str(cm[i, j]), ha="center", va="center")

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        axes[1].imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        axes[1].set_title("Confusion Matrix (Normalized)")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

        for ax in axes:
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
        plt.tight_layout()

        cm_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Confusion matrix saved to {cm_path}")

    def _plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float):
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        roc_path = self.output_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ROC curve saved to {roc_path}")

    def _plot_sample_windows_by_outcome(
        self, X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ):
        """Plot one sample window for each outcome (TP, TN, FP, FN)."""
        outcomes = {
            "TP": (y_test == 1) & (y_pred == 1),
            "TN": (y_test == 0) & (y_pred == 0),
            "FP": (y_test == 0) & (y_pred == 1),
            "FN": (y_test == 1) & (y_pred == 0),
        }

        selected = []
        labels = []
        for name, mask in outcomes.items():
            indices = np.where(mask)[0]
            if indices.size > 0:
                idx = int(indices[0])
                selected.append(idx)
                labels.append(name)

        if not selected:
            logger.warning("No sample windows available for outcome plotting.")
            return

        fig, axes = plt.subplots(len(selected), 1, figsize=(12, 3 * len(selected)))
        if len(selected) == 1:
            axes = [axes]

        for ax, idx, label in zip(axes, selected, labels):
            window = X_test[idx]
            timesteps = np.arange(len(window))
            for feature_idx in range(window.shape[1]):
                ax.plot(
                    timesteps,
                    window[:, feature_idx],
                    label=self.config.feature_columns[feature_idx],
                    alpha=0.7,
                )

            true_label = int(y_test[idx])
            pred_label = int(y_pred[idx])
            pred_prob = float(y_pred_proba[idx])
            ax.set_title(
                f"{label}: true={true_label}, pred={pred_label}, prob={pred_prob:.3f}"
            )
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Normalized Value")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        sample_path = self.output_dir / "sample_windows.png"
        plt.savefig(sample_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Sample window plot saved to {sample_path}")

    def plot_training_history(self):
        """Plot training history."""
        history_path = self.output_dir / "training_history.json"
        if not history_path.exists():
            logger.warning(f"Training history not found: {history_path}")
            return

        with open(history_path) as f:
            history = json.load(f)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        axes[0].plot(history.get("loss", []), label="Train Loss")
        axes[0].plot(history.get("val_loss", []), label="Val Loss")
        axes[0].set_title("Loss vs Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history.get("accuracy", []), label="Train Accuracy")
        axes[1].plot(history.get("val_accuracy", []), label="Val Accuracy")
        axes[1].set_title("Accuracy vs Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(history.get("precision", []), label="Train Precision")
        axes[2].plot(history.get("val_precision", []), label="Val Precision")
        axes[2].set_title("Precision vs Epoch")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Precision")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(history.get("recall", []), label="Train Recall")
        axes[3].plot(history.get("val_recall", []), label="Val Recall")
        axes[3].set_title("Recall vs Epoch")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Recall")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        history_path_plot = self.output_dir / "training_curves.png"
        plt.savefig(history_path_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Training curves saved to {history_path_plot}")
