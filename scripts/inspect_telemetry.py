"""Inspect the structure of telemetry data from dataset v1."""
import json
from pathlib import Path
import pandas as pd

dataset_dir = Path("data/dataset_v1")

# Get first JSON metadata file
json_files = sorted(dataset_dir.glob("*.json"))
if not json_files:
    print("No JSON files found!")
    exit()

first_json = json_files[0]
print(f"Inspecting: {first_json.name}\n")

with open(first_json, 'r') as f:
    metadata = json.load(f)

print("Metadata keys:")
for key, value in metadata.items():
    if isinstance(value, (str, int, float)):
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: {type(value)}")

# Find telemetry directory
output_dir = Path(metadata.get('output_directory'))
print(f"\nOutput directory: {output_dir}")
print(f"Exists: {output_dir.exists()}")

if output_dir.exists():
    # Look for CSV files recursively
    csv_files = sorted(output_dir.glob("*.csv"))
    print(f"\nCSV files in root: {len(csv_files)}")
    
    # Look for subdirectories
    subdirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    print(f"Subdirectories: {len(subdirs)}")
    if subdirs:
        print(f"  First few: {[d.name for d in subdirs[:5]]}")
    
    # Search recursively for CSV files
    all_csv_files = sorted(output_dir.glob("**/*.csv"))
    print(f"\nCSV files (recursive search): {len(all_csv_files)}")
    
    if all_csv_files:
        first_csv = all_csv_files[0]
        print(f"\nInspecting first CSV: {first_csv}")
        print(f"Relative path: {first_csv.relative_to(dataset_dir)}")
        
        try:
            df = pd.read_csv(first_csv, nrows=5)
            print(f"\nShape: {df.shape}")
            print(f"\nColumns: {df.columns.tolist()}")
            print(f"\nFirst few rows:\n{df}")
            print(f"\nData types:\n{df.dtypes}")
        except Exception as e:
            print(f"Error reading CSV: {e}")
    
    # Also check what's in the metadata about fault_distribution
    print(f"\n\nFault distribution:")
    print(metadata.get('fault_distribution'))