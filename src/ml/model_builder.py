import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)

def build_lstm(window_size: int, num_features: int, num_classes: int) -> keras.Model:
    """
    Build LSTM-based fault detection model.
    
    Args:
        window_size: Number of timesteps per window
        num_features: Number of input features
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(window_size, num_features)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        layers.LSTM(32, dropout=0.2),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
    ])
    
    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
    
    logger.info(f"Built LSTM model for {num_classes} classes")
    return model


def build_cnn1d(window_size: int, num_features: int, num_classes: int) -> keras.Model:
    """
    Build 1D CNN-based fault detection model.
    
    Args:
        window_size: Number of timesteps per window
        num_features: Number of input features
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(window_size, num_features)),
        
        # Conv block 1
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling1D(pool_size=2),
        
        # Conv block 2
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling1D(pool_size=2),
        
        # Conv block 3
        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
    ])
    
    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
    
    logger.info(f"Built CNN1D model for {num_classes} classes")
    return model


def build_model(config, num_classes: int) -> keras.Model:
    """
    Factory function to build model based on config.
    
    Args:
        config: Phase4Config instance
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    if config.model_type == "lstm":
        model = build_lstm(config.window_size, config.num_features, num_classes)
    elif config.model_type == "cnn1d":
        model = build_cnn1d(config.window_size, config.num_features, num_classes)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    # Compile
    if num_classes == 2:
        loss = "binary_crossentropy"
        metrics = ["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
    else:
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    logger.info(f"Model compiled with {loss}")
    return model