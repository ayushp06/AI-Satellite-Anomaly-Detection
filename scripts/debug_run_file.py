"""Debug script to inspect raw run file structure."""
import json
from pathlib import Path
import pandas as pd

dataset_dir = Path("data/dataset_v1")
telemetry_file = sorted(dataset_dir.glob("telemetry_run_*.parquet"))[0]
run_id = telemetry_file.stem.split("_")[-1]
metadata_file = dataset_dir / f"metadata_run_{run_id}.json"

print(f"Telemetry file: {telemetry_file.name}")
print(f"Telemetry size: {telemetry_file.stat().st_size / 1024:.1f} KB\n")

df = pd.read_parquet(telemetry_file)
print("Telemetry columns:", list(df.columns))
print("Telemetry head:")
print(df.head())

if metadata_file.exists():
    print(f"\nMetadata file: {metadata_file.name}")
    print(f"Metadata size: {metadata_file.stat().st_size / 1024:.1f} KB\n")

    with open(metadata_file, "r") as f:
        run_data = json.load(f)

    print("Metadata keys:")
    for key in run_data.keys():
        value = run_data[key]
        if isinstance(value, list):
            print(f"  {key}: list[{len(value)}]")
            if len(value) > 0:
                print(f"    - First item type: {type(value[0])}")
                if isinstance(value[0], dict):
                    print(f"    - First item keys: {list(value[0].keys())}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with keys {list(value.keys())}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")

    print("\nfault_windows detail:")
    print(json.dumps(run_data.get("fault_windows"), indent=2))
else:
    print("\nNo metadata file found for this run.")
