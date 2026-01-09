# Phase 4: Fault Detection (Corrected)

## Summary

A degenerate classifier was observed during initial Phase 4 evaluation: the model predicted every window as a fault. This document explains why that happened, the corrective changes made to the window-labeling and evaluation pipeline, and how to interpret the updated outputs.

## Root Cause

The initial window labeling used an aggressive rule (e.g., `max(label)`), which marks a window as faulty if any timestep in the window is faulty. With stride 1 and 5-second windows, this contaminated many windows that were mostly nominal. As a result, the training set became heavily biased toward the fault class, and the model learned to predict faults for all windows. This also introduced temporal leakage: labels were influenced by future samples within the same window.

## Fixes Implemented

1. Window labeling refactor (leakage-safe)
   - `center` (default): label = center timestep of the window.
   - `percentage`: label = 1 only if the fraction of fault timesteps is >= threshold (default 0.2).

2. Dataset integrity checks
   - Class distribution logged before and after windowing.
   - Training aborts if any class exceeds 95% of samples.
   - Train/val/test split is chronological (time-based).

3. Imbalance handling
   - Class-weighted loss is computed from training labels.

4. Evaluation overhaul

Notes:
- Default features in the code are `q0..q3` and `w0..w2` (from the parquet telemetry).
- If your source uses `wx, wy, wz`, the pipeline will alias them automatically or you can set `feature_columns` in `config/phase4_config.py`.

   - Raw + normalized confusion matrix
   - Precision / Recall / F1
   - ROC curve + AUC
   - Representative windows for TP/TN/FP/FN with predicted probabilities

## Artifact Outputs

Artifacts are written to the configured `output_dir` (default `results/phase4/`):

- `model.keras`
- `scaler.joblib`
- `metrics.json`
- `confusion_matrix.png`
- `roc_curve.png`
- `training_curves.png`
- `sample_windows.png`
- `config_used.json`

## Results (Latest Run)

The latest metrics on disk are shown below. Re-run training after the fixes to refresh these numbers.

```json
{
  "test_loss": 4.160618782043457,
  "num_classes": 2,
  "model_type": "lstm",
  "window_size": 50,
  "batch_size": 32,
  "epochs_trained": 12,
  "test_accuracy": 0.2689056992530823,
  "test_precision": 0.2689056992530823,
  "test_recall": 1.0
}
```

## Lessons Learned

- Window labeling must avoid future information leakage; center labeling is the safest default.
- Aggressive window labeling inflates the fault class and can yield degenerate classifiers.
- Always validate class distribution before training and after windowing.
- Report probability scores and multiple metrics (Precision/Recall/AUC) to catch failure modes.


## Temporal Class Collapse in Time-Series Evaluation

A naive chronological split can allocate all rare fault events to the early portion of the timeline. When that happens, the test set contains only nominal samples, which makes ROC/AUC undefined and hides false negatives. The model may appear to perform well while the evaluation is no longer meaningful for fault detection.

To prevent this, Phase 4 uses block-aware time splitting. Contiguous fault and nominal runs are grouped into blocks, and splits are performed on block boundaries. This preserves temporal order within each block and ensures that each split contains both classes. Windowing is applied after splitting, and normalization is fit only on the training split to avoid leakage.

For spacecraft telemetry, this matters because faults are often rare and clustered. If the evaluation split lacks faults, operational recall cannot be estimated. Block-aware splitting provides a causal, time-respecting evaluation that remains statistically valid for anomaly detection.
