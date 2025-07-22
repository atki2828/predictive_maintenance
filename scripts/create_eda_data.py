import os

import polars as pl

from app.pmhelpers.dataprocessing import (
    create_fail_plot_df,
    create_main_fail_flag_df,
    create_time_between_fail_group_df,
    create_time_to_fail_df,
    event_pivoter,
)

TELEMETRY_FILE_PATH = "./data/PdM_telemetry.csv"
FAILURE_FILE_PATH = "./data/PdM_failures.csv"
MACHINE_FILE_PATH = "./data/PdM_machines.csv"
MAINTENANCE_FILE_PATH = "./data/PdM_maintenance.csv"
ERROR_FILE_PATH = "./data/PdM_errors.csv"
WRITE_DIR = "./data"


def main():
    # Read in Raw Data
    df_tel = pl.read_csv(TELEMETRY_FILE_PATH, try_parse_dates=True)
    df_err = pl.read_csv(ERROR_FILE_PATH, try_parse_dates=True)
    df_fail = pl.read_csv(FAILURE_FILE_PATH, try_parse_dates=True)
    df_mach = pl.read_csv(MACHINE_FILE_PATH, try_parse_dates=True)
    df_main = pl.read_csv(MAINTENANCE_FILE_PATH, try_parse_dates=True)

    comp_fail_plot_df = create_fail_plot_df(df_fail=df_fail, fail_col="failure")
    mach_fail_plot_df = create_fail_plot_df(df_fail=df_fail, fail_col="machineID")

    time_to_fail_df = create_time_to_fail_df(
        df_fail=df_fail, window_cols=["machineID", "failure"]
    )

    main_fail_flag_df = create_main_fail_flag_df(df_main=df_main, df_fail=df_fail)
    max_datetime = df_tel.select(pl.col("datetime").max()).to_series()[0]
    time_between_fail_flag_df = create_time_between_fail_group_df(
        fail_flag_df=main_fail_flag_df,
        window_cols=["machineID", "comp"],
        max_datetime=max_datetime,
    ).join(df_mach, on="machineID", how="inner")

    # The next steps create the daily agg df
    daily_agg_df = (
        df_tel.with_columns(pl.col("datetime").dt.date().alias("date"))
        .group_by(["machineID", "date"])
        .agg(
            [
                pl.col("volt").mean().alias("mean_daily_voltage"),
                pl.col("rotate").mean().alias("mean_daily_rotation"),
                pl.col("pressure").mean().alias("mean_daily_pressure"),
                pl.col("vibration").mean().alias("mean_daily_vibration"),
            ]
        )
    )

    daily_fail_pivot_df = event_pivoter(df_fail, "failure")
    daily_error_pivot_df = event_pivoter(df_err, "errorID")
    daily_maint_pivot_df = event_pivoter(df_main, "comp")

    daily_plot_df = (
        daily_agg_df.join(daily_error_pivot_df, on=["machineID", "date"], how="left")
        .join(daily_maint_pivot_df, on=["machineID", "date"], how="left")
        .join(
            daily_fail_pivot_df, on=["machineID", "date"], how="left", suffix="_failure"
        )
        .fill_null(0)
        .sort(["machineID", "date"])
    )

    comp_fail_plot_df.write_csv(os.path.join(WRITE_DIR, "comp_fail_plot.csv"))
    mach_fail_plot_df.write_csv(os.path.join(WRITE_DIR, "machine_fail_plot.csv"))
    time_to_fail_df.write_csv(os.path.join(WRITE_DIR, "time_to_fail.csv"))
    time_between_fail_flag_df.write_csv(
        os.path.join(WRITE_DIR, "time_between_fail_flag.csv")
    )
    daily_plot_df.write_csv(os.path.join(WRITE_DIR, "daily_plot_df.csv"))


if __name__ == "__main__":
    main()
