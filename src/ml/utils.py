import numpy as np
import tensorflow as tf
from pathlib import Path

def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_model(model_path: Path) -> tf.keras.Model:
    """Load saved model."""
    return tf.keras.models.load_model(str(model_path))


def predict_on_data(model: tf.keras.Model, X: np.ndarray, 
                   num_classes: int) -> np.ndarray:
    """
    Make predictions on data.
    
    Args:
        model: Keras model
        X: Input features (batch_size, window_size, num_features)
        num_classes: Number of output classes
    
    Returns:
        Predicted class labels
    """
    y_pred_proba = model.predict(X)
    
    if num_classes == 2:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_pred