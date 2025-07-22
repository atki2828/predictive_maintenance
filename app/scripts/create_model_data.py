import os

import polars as pl

from app.pmhelpers.dataprocessing import (
    create_anomaly_df,
    create_daily_component_fail_df,
    create_fail_prejoin_df,
    create_main_prejoin_df,
)

# This daily_plot_df.csv needs to be created by running create_eda_data.py
DAILY_PLOT_DATA_PATH = "./data/daily_plot_df.csv"
TELEMETRY_FILE_PATH = "./data/PdM_telemetry.csv"
FAILURE_FILE_PATH = "./data/PdM_failures.csv"
MACHINE_FILE_PATH = "./data/PdM_machines.csv"
MAINTENANCE_FILE_PATH = "./data/PdM_maintenance.csv"
ERROR_FILE_PATH = "./data/PdM_errors.csv"
WRITE_PATH = "./data/"
mean_telemetry_cols = [
    "mean_daily_voltage",
    "mean_daily_rotation",
    "mean_daily_pressure",
    "mean_daily_vibration",
]

window_size = 21  # In days
z_anomaly_threshold = 2.3  # z threshold for deciding anomaly


def main():
    df_tel = pl.read_csv(TELEMETRY_FILE_PATH, try_parse_dates=True)
    df_main = pl.read_csv(MAINTENANCE_FILE_PATH, try_parse_dates=True)
    df_fail = pl.read_csv(FAILURE_FILE_PATH, try_parse_dates=True)
    df_mach = pl.read_csv(MACHINE_FILE_PATH, try_parse_dates=True)
    # This should be renamed to daily agg, but keeping for consistency
    daily_plot_df = pl.read_csv(DAILY_PLOT_DATA_PATH, try_parse_dates=True)
    daily_telemetry_z_df = create_anomaly_df(
        df=daily_plot_df,
        mean_telemetry_cols=mean_telemetry_cols,
        window_size=window_size,
        z_anomaly_threshold=z_anomaly_threshold,
    )

    # Create maintenance prejoin df
    start_date = df_main.select(
        pl.col("datetime").dt.date().min().alias("start_date")
    ).item()
    end_date = df_tel.select(
        pl.col("datetime").dt.date().max().alias("end_date")
    ).item()
    main_prejoin_df = create_main_prejoin_df(
        df=df_main, start_date=start_date, end_date=end_date
    )
    # Create Fail Prejoin Df
    fail_prejoin_df = create_fail_prejoin_df(df_fail)

    daily_component_fail_df = create_daily_component_fail_df(
        main_prejoin_df=main_prejoin_df, fail_prejoin_df=fail_prejoin_df
    )

    # Get date with earliest telemetry data
    earliest_telem_date = df_tel.select(
        pl.col("datetime").dt.date().min().alias("end_date")
    ).item()
    # Create pre telemetry component instance ids
    pre_telem_component_instance_list = (
        daily_component_fail_df.filter(pl.col("date") < earliest_telem_date)
        .select("component_instance_id")
        .to_series()
        .unique()
    )
    # Filter out component instance ids with no telemetry data
    daily_component_fail_df_filtered = daily_component_fail_df.filter(
        ~pl.col("component_instance_id").is_in(pre_telem_component_instance_list)
    )

    model_df = daily_component_fail_df_filtered.join(
        daily_telemetry_z_df, on=["date", "machineID"]
    ).join(df_mach, on="machineID")

    model_df.write_csv(os.path.join(WRITE_PATH, "model_data.csv"))


if __name__ == "__main__":
    main()
