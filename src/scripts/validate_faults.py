import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = ["t", "w0", "w1", "w2", "fault"]


def _fault_segments(df):
    fault_idx = df.index[df["fault"] == 1].to_list()
    if not fault_idx:
        return []

    segments = []
    start = fault_idx[0]
    prev = fault_idx[0]
    for idx in fault_idx[1:]:
        if idx != prev + 1:
            segments.append((start, prev))
            start = idx
        prev = idx
    segments.append((start, prev))
    return segments


def _summarize_anomalies(df):
    total = len(df)
    fault_mask = df["fault"] == 1
    fault_points = int(fault_mask.sum())
    ratio = fault_points / total if total else 0.0

    print(f"Rows: {total}")
    print(f"Fault points: {fault_points} ({ratio:.2%})")

    segments = _fault_segments(df)
    if segments:
        print(f"Fault segments: {len(segments)}")
        for i, (start, end) in enumerate(segments, start=1):
            start_t = df.at[start, "t"]
            end_t = df.at[end, "t"]
            duration = end_t - start_t
            count = end - start + 1
            print(f"  Segment {i}: t={start_t:.3f}..{end_t:.3f} ({duration:.3f}s, {count} rows)")
    else:
        print("Fault segments: 0")

    normal = df.loc[~fault_mask, ["w0", "w1", "w2"]]
    fault = df.loc[fault_mask, ["w0", "w1", "w2"]]

    if fault.empty:
        print("No faults found; cannot compare anomaly magnitude.")
        return

    normal_mean = normal.mean()
    normal_std = normal.std().replace(0, pd.NA)

    print("Normal stats (mean/std):")
    for col in ["w0", "w1", "w2"]:
        print(f"  {col}: {normal_mean[col]:.6f} / {normal_std[col]:.6f}")

    print("Fault stats (mean/std):")
    for col in ["w0", "w1", "w2"]:
        print(f"  {col}: {fault[col].mean():.6f} / {fault[col].std():.6f}")

    obvious = False
    print("Max abs z-score during faults (vs normal mean/std):")
    for col in ["w0", "w1", "w2"]:
        std = normal_std[col]
        if pd.isna(std):
            print(f"  {col}: std=0 in normal data; cannot compute z-score.")
            continue
        max_fault_z = ((fault[col] - normal_mean[col]).abs() / std).max()
        max_normal_z = ((normal[col] - normal_mean[col]).abs() / std).max()
        print(f"  {col}: fault max z={max_fault_z:.2f}, normal max z={max_normal_z:.2f}")
        if max_fault_z >= 3 and max_fault_z >= max_normal_z * 1.5:
            obvious = True

    if not obvious:
        print("Warning: faults are not strongly separated from normal behavior.")

    if ratio > 0.2:
        print("Warning: fault coverage is high; anomalies may not be isolated.")
    elif len(segments) > 5:
        print("Warning: many fault segments; anomalies may not be isolated.")


def _plot_faults(df, output_path, show):
    t = df["t"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, df["w0"], label="w0")
    ax.plot(t, df["w1"], label="w1")
    ax.plot(t, df["w2"], label="w2")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angular velocity (rad/s)")

    ax2 = ax.twinx()
    ax2.step(t, df["fault"], where="post", color="black", alpha=0.35, label="fault")
    ax2.set_ylabel("fault flag")
    ax2.set_ylim(-0.05, 1.05)

    for start, end in _fault_segments(df):
        ax.axvspan(df.at[start, "t"], df.at[end, "t"], color="red", alpha=0.12)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    ax.set_title("Angular velocity with fault overlay")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Validate fault visibility and isolation.")
    parser.add_argument("--input", default="telemetry.parquet", help="Path to telemetry parquet.")
    parser.add_argument(
        "--output",
        default="plots/faults_overlay.png",
        help="Path to save the output plot.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not df["t"].is_monotonic_increasing:
        df = df.sort_values("t").reset_index(drop=True)
        print("Note: timestamps were not monotonic; sorted by time.")

    _summarize_anomalies(df)
    _plot_faults(df, Path(args.output), args.show)


if __name__ == "__main__":
    main()
