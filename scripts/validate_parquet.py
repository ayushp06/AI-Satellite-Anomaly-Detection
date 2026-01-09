import pandas as pd

# These are the columns we expect to see in telemetry.parquet
# The order matters because our schema is fixed
EXPECTED_COLUMNS = [
    "t",
    "q0", "q1", "q2", "q3",
    "w0", "w1", "w2",
    "fault"
]

# Load the telemetry file
df = pd.read_parquet("telemetry.parquet")

# Print the columns so we can visually confirm them
print("Columns:", list(df.columns))

# Make sure the columns match exactly what we expect
assert list(df.columns) == EXPECTED_COLUMNS, "Column names or order are wrong"

# Make sure there are no missing values anywhere in the file
assert df.isna().sum().sum() == 0, "There are missing (NaN) values in the data"

# The fault column should only contain 0 or 1
assert df["fault"].isin([0, 1]).all(), "Fault column contains invalid values"

# Time should never be negative
assert (df["t"] >= 0).all(), "Time contains negative values"

# Make sure the file actually has data in it
assert len(df) > 0, "Telemetry file is empty"

# Print a few rows so we can inspect the data manually
print(df.head())

# Print how many rows were logged
print(f"Rows: {len(df)}")

# If we reach this point, everything passed
print("OK: Phase 1 telemetry is valid")
