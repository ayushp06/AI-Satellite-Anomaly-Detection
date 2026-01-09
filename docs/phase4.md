# Phase 4: Fault Detection ML Pipeline

## Overview

Phase 4 implements a TensorFlow/Keras time-series machine learning pipeline for detecting and classifying faults in spacecraft attitude telemetry data produced by Phase 3.

### Features

- **Data Pipeline**: Sliding window creation, normalization, chronological train/val/test split
- **Model Architectures**: LSTM and 1D CNN options
- **Training**: Comprehensive callbacks (checkpoint, early stopping, LR reduction)
- **Evaluation**: Confusion matrices, classification reports, sample prediction plots
- **Reproducibility**: Random seed control, scaler serialization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Run training with default config:

```bash
python scripts/train_fault_detector.py
```

With custom parameters:

```bash
python scripts/train_fault_detector.py \
  --data data/phase3_dataset.csv \
  --model lstm \
  --window-seconds 5.0 \
  --batch-size 32 \
  --epochs 100 \
  --learning-rate 0.001
```

Available models: `lstm`, `cnn1d`

### Evaluation Only

If you've already trained a model:

```bash
python scripts/train_fault_detector.py --eval-only
```

## Configuration

Edit `config/phase4_config.py` to modify:

- **Data paths**: `data_path`, `output_dir`
- **Window parameters**: `window_seconds`, `sample_rate_hz`, `stride`
- **Features**: `feature_columns` (quaternion + angular velocity)
- **Splits**: `test_split`, `val_split` (chronological order)
- **Hyperparameters**: `batch_size`, `epochs`, `learning_rate`
- **Model**: `model_type` (lstm/cnn1d), layer sizes, dropout
- **Preprocessing**: `normalize_quaternions` (normalize to unit length)

## Artifacts

Training/evaluation generates in `results/phase4/`:

```
results/phase4/
├── best_model.keras          # Best model (by val loss)
├── scaler.pkl                # StandardScaler (fit on train)
├── config.json               # Config used
├── metrics.json              # Test metrics
├── training_history.json     # Per-epoch loss/accuracy
├── classification_report.json# Precision/recall/F1
└── plots/
    ├── confusion_matrix.png
    ├── sample_predictions.png
    └── training_history.png
```

## Dataset Format

Input CSV must have:

- **Timestamp column** (named: `timestamp`, `t`, or `time`) — for sorting
- **Features**: `q0, q1, q2, q3` (quaternion), `w0, w1, w2` (angular velocity)
- **Label**: Integer column named `fault` (0=nominal, 1=fault)

Example:
```
timestamp,q0,q1,q2,q3,w0,w1,w2,fault
0.0,0.999,-0.001,0.001,0.002,0.01,-0.005,0.002,0
0.1,0.999,-0.001,0.001,0.002,0.01,-0.005,0.002,0
```

## Models

### LSTM
- Masking + 2 LSTM layers (64, 32 units)
- Dropout (0.2, 0.3)
- Dense head (16 units)
- Binary sigmoid or softmax output

### CNN1D
- 3 Conv1D blocks (32, 64, 128 filters)
- Batch normalization + dropout (0.2, 0.3)
- Global average pooling
- Dense head (64 units)
- Binary sigmoid or softmax output

## Project Structure

```
AI-Satellite-Anomaly-Detection/
├── config/
│   └── phase4_config.py          # Phase 4 configuration
├── src/ml/
│   ├── data_pipeline.py          # Data loading & windowing
│   ├── model_builder.py          # LSTM/CNN1D builders
│   ├── trainer.py                # Training loop
│   ├── evaluator.py              # Evaluation & plotting
│   └── utils.py                  # Utility functions
├── scripts/
│   └── train_fault_detector.py   # CLI entry point
├── results/
│   └── phase4/                   # Artifacts & plots
└── docs/
    └── phase4.md                 # This file
```