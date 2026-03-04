import pandas as pd

# Load CSV
df = pd.read_csv("mock_imu_gps.csv")

# Filter out rows where all EUL values are 0
filtered_df = df[~(
    (df["GPSNavEULX"] == 0) &
    (df["GPSNavEULY"] == 0) &
    (df["GPSNavEULZ"] == 0)
)]

# Save result
filtered_df.to_csv("filtered_mock_imu_gps.csv", index=False)

print("Original rows:", len(df))
print("Filtered rows:", len(filtered_df))