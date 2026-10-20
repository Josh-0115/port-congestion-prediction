
from pathlib import Path

import numpy as np
import pandas as pd

# Reproducible synthetic data
rng = np.random.default_rng(42)
N_RECORDS = 300

# Simulated port operating conditions
arrival_density = rng.integers(5, 46, N_RECORDS)
port_capacity = rng.integers(15, 36, N_RECORDS)
queue_length = rng.integers(0, 26, N_RECORDS)
weather_severity = rng.integers(0, 4, N_RECORDS)
historical_avg_delay = rng.uniform(1, 20, N_RECORDS).round(2)
hour_of_day = rng.integers(0, 24, N_RECORDS)

seasons = np.array(["winter", "spring", "summer", "autumn"])
season = rng.choice(seasons, N_RECORDS)

# Additional pressure when vessel arrivals exceed capacity
over_capacity = np.maximum(
    arrival_density - port_capacity, 0
)

# Illustrative simulation rule, not a real-world delay formula
rush_hour = np.isin(hour_of_day, [7, 8, 9, 17, 18, 19]).astype(int)

noise = rng.normal(0, 3, N_RECORDS)

delay_hours = (
    1
    + 0.45 * queue_length
    + 0.55 * over_capacity
    + 2.0 * weather_severity
    + 0.25 * historical_avg_delay
    + 1.5 * rush_hour
    + noise
)

delay_hours = np.maximum(delay_hours, 0).round(2)

# Our chosen definition: significant delay > 12 hours
congestion_label = (delay_hours > 12).astype(int)

df = pd.DataFrame({
    "arrival_density": arrival_density,
    "port_capacity": port_capacity,
    "queue_length": queue_length,
    "weather_severity": weather_severity,
    "historical_avg_delay": historical_avg_delay,
    "hour_of_day": hour_of_day,
    "season": season,
    "delay_hours": delay_hours,
    "congestion_label": congestion_label,
})

output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "synthetic_ports.csv"
df.to_csv(output_file, index=False)

print(f"Dataset saved to: {output_file}")
print(f"Number of records: {len(df)}")
print("\nFirst five records:")
print(df.head().to_string(index=False))

print("\nCongestion label distribution:")
print(df["congestion_label"].value_counts().sort_index())
