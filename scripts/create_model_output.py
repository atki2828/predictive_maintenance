import polars as pl

MEAN_TELEMETRY_COLS = [
    "mean_daily_voltage",
    "mean_daily_rotation",
    "mean_daily_pressure",
    "mean_daily_vibration",
]

window_size = 21  # In days
z_anomaly_threshold = 2.3  # z threshold for deciding anomaly
