# ML Plan — TensorFlow Satellite Telemetry Anomaly Detection

## Problem Statement
Given multivariate time-series telemetry (q, w), detect anomalies (faults) in near-real-time and offline.

We will start with simulation-generated datasets with injected faults and a `fault` label.
The ML model will be trained primarily on normal behavior and used to score deviations.

## Data
### Signals (base)
- q0, q1, q2, q3 (quaternion)
- w0, w1, w2 (angular velocity)
- t (timestamp)
- fault (0/1 label for evaluation and for selecting normal training regions)

### Derived Features (optional; add after baseline works)
- |w| = sqrt(w0^2 + w1^2 + w2^2)
- dq_norm = |q(t) - q(t-1)| (or quaternion error measure)
- rolling statistics: rolling mean/std of w channels
Note: keep derived features deterministic and logged or recomputable.

## Sampling / Windowing
- Simulation rate: dt = 0.1s (10Hz) recommended
- Window length: start with 128 timesteps (~12.8 seconds)
- Stride: 1 window per timestep (for best detection) OR stride=8/16 for faster training
- Window tensor shape: (T, F)
  - T = 128
  - F = 7 (q0..q3, w0..w2) OR more if derived features included

## Train/Val/Test Split
Split by time (no random shuffling across time):
- Train: earlier segment, primarily normal
- Val: middle segment, mixed normal/fault for threshold tuning
- Test: later segment, mixed normal/fault for final metrics

## Model Choice (Phase 4)
### Primary: LSTM Autoencoder (recommended first)
- Encoder: 1–2 LSTM layers reduce sequence to latent vector
- Decoder: repeats latent and decodes back to (T, F)
- Loss: mean squared error (MSE) reconstruction loss over the window

Why:
- Works well for telemetry-type sequences
- Unsupervised/semi-supervised: can train on normal data
- Reconstruction error provides interpretable anomaly score

### Alternative models (future)
- TCN (Temporal Convolutional Network) autoencoder
- Transformer-based forecasting (score = prediction residual)
- Variational autoencoder (VAE) for probabilistic scoring

## Training Strategy
- Train using windows where `fault=0` (normal-only)
- Validate with both normal and fault windows to choose a threshold
- Save artifacts:
  - trained TF model (`.keras`)
  - scaler statistics (mean/std) as JSON
  - chosen threshold as JSON

## Scoring
For each window:
- reconstructed_window = model(window)
- error = mean((window - reconstructed_window)^2) over (T,F)
- produce a scalar anomaly score per window
Optionally also track per-feature reconstruction error for interpretability.

## Thresholding
Choose threshold using validation set:
- percentile method: threshold = 99.5th percentile of normal scores in validation
- or maximize F1 score vs fault labels (since in sim we have labels)

## Event Detection (turn scores into incidents)
- Smooth score with EMA or moving average
- Trigger event when score > threshold for N consecutive windows
- End event after score stays below threshold for M consecutive windows
- Merge events within gap <= G seconds
Output:
- event list: start_t, end_t, peak_score, duration, optional dominant feature

## Metrics
Because simulation provides `fault`:
- Precision, Recall, F1 (event-based and pointwise)
- False alarms per hour
- Detection delay (time between fault start and first detection)
- ROC-AUC on window labels (optional)

## Deliverables by Phase
Phase 3:
- dataset builder + windowing + scaling + saved artifacts

Phase 4:
- TensorFlow model training script
- saved model + scaler + threshold

Phase 5:
- detector script for:
  - offline scoring on a Parquet file
  - real-time scoring during simulation

Phase 6:
- plots + report + optional PySide6 UI
