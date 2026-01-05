# AI Satellite Anomaly Detection — Architecture

## Goal
Build an end-to-end anomaly detection system for satellite-like telemetry.

Primary mode (now): simulation-generated telemetry (attitude dynamics) with injected faults.
Secondary mode (later): real telemetry ingestion (CSV/Parquet) with the same ML pipeline.

## Current State (as of Phase 1)
We have:
- Attitude simulator that updates:
  - quaternion q = [q0,q1,q2,q3]
  - angular velocity w = [w0,w1,w2]
- Telemetry builder that packages time-series records
- Telemetry logger that writes a Parquet time-series dataset

Target telemetry table schema:
t, q0, q1, q2, q3, w0, w1, w2, fault

## System Blocks (Pipeline)
1) Simulation / Ingestion
   - Simulation generates "truth" q and w at fixed dt (e.g., 0.1s = 10Hz)
   - Later, ingestion reads real telemetry logs

2) Fault Injection (Phase 2)
   - Create controlled anomalies to produce labeled data
   - Examples:
     - gyro bias step
     - noise burst / spikes
     - sensor dropout (freeze or NaNs)
     - torque disturbance event

3) Telemetry Logging (Phase 1)
   - Streaming-like telemetry is buffered in memory and flushed to Parquet in batches
   - Ensures a stable dataset format for ML

4) Preprocessing (Phase 3)
   - Sort by time, resample to uniform dt if needed
   - Handle missing values
   - Scale channels (fit scaler on TRAIN only)

5) Windowing (Phase 3)
   - Convert time-series into fixed-length windows for ML
   - Example:
     - sample_rate = 10Hz
     - window_len = 128 timesteps (~12.8s)
     - features = [q0..q3, w0..w2] (+ optional derived features)

6) Model Training (Phase 4)
   - TensorFlow anomaly detector (initially LSTM autoencoder)
   - Train mostly/only on "normal" windows (fault=0)

7) Scoring + Event Detection (Phase 5)
   - Score each timestep/window using reconstruction error
   - Apply smoothing + thresholding + minimum duration
   - Output anomaly events: (start_t, end_t, severity)

8) Reporting + Visualization (Phase 6)
   - Metrics: false alarms/hr, precision/recall, detection delay
   - Plot telemetry channels + anomaly score overlay
   - UI (PySide6) for live view

## Interfaces (What each block should expose)
- Simulation runner:
  - output: streaming telemetry dicts or rows
- Logger:
  - input: single telemetry row (already flattened or flattenable)
  - output: Parquet file
- Dataset builder:
  - input: Parquet
  - output: numpy arrays or TF datasets of windows
- Model:
  - input: window tensor (T x F)
  - output: reconstructed window (T x F) or predicted next value(s)
- Detector:
  - input: telemetry stream or file
  - output: anomaly score series + event list

## Design Rules
- Keep telemetry schema stable (single source of truth)
- Separate simulation/faults from ML code
- Ensure preprocessing/scaling is reproducible and saved as artifacts
- Split train/val/test by time to avoid leakage
