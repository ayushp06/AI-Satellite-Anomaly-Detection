"""Inspect the structure of telemetry data from dataset v1."""
import json
from pathlib import Path
import pandas as pd

dataset_dir = Path("data/dataset_v1")

telemetry_files = sorted(dataset_dir.glob("telemetry_run_*.parquet"))
if not telemetry_files:
    print("No telemetry_run_*.parquet files found!")
    exit()

print(f"Found {len(telemetry_files)} telemetry files\n")

telemetry_file = telemetry_files[0]
run_id = telemetry_file.stem.split("_")[-1]
metadata_file = dataset_dir / f"metadata_run_{run_id}.json"

print(f"{'='*60}")
print(f"INSPECTING: {telemetry_file.name}")
print(f"{'='*60}\n")

if metadata_file.exists():
    with open(metadata_file, "r") as f:
        run_data = json.load(f)

    print("Run metadata keys:")
    for key, value in run_data.items():
        if isinstance(value, (str, int, float, bool)):
            print(f"  {key}: {value}")
        elif isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
        else:
            print(f"  {key}: {type(value)}")
else:
    run_data = {}
    print("No metadata file found for this run.")

print(f"\n{'='*60}")
print("DATAFRAME STRUCTURE")
print(f"{'='*60}\n")

df = pd.read_parquet(telemetry_file)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:")
print(df.dtypes)
print(f"\nFirst 5 rows:")
print(df.head())

if run_data.get("fault_windows"):
    print(f"\n{'='*60}")
    print("FAULT WINDOWS")
    print(f"{'='*60}")
    print(f"fault_injected: {run_data.get('fault_injected')}")
    print(f"fault_windows: {run_data.get('fault_windows')}")
