from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st


@st.cache_data
def load_data(file_path: str) -> pl.DataFrame:
    df = pl.read_csv(file_path, try_parse_dates=True)
    return df


def create_time_between_fail_group_df(
    fail_flag_df: pl.DataFrame, window_cols: list[str], max_datetime
) -> pl.DataFrame:
    time_between_fail_group_df = (
        fail_flag_df.with_columns(
            [
                pl.col("datetime")
                .shift(-1)
                .over(window_cols)
                .alias("next_replace_time"),
                pl.col("fail_flag").shift(-1).over(window_cols).fill_null(0),
            ]
        )
        .with_columns(
            [
                pl.col("next_replace_time").fill_null(max_datetime),
                (pl.col("next_replace_time") - pl.col("datetime"))
                .dt.total_days()
                .alias("time_between_maintenance"),
            ]
        )
        .filter(pl.col("time_between_maintenance") > 0)
    )

    return time_between_fail_group_df


def get_failures_by_comp(df_fail: pl.DataFrame, machine_id: int, comp: str):
    return df_fail.filter(pl.col("machineID") == machine_id).filter(
        pl.col("comp") == comp
    )


def get_component_failure(df: pd.DataFrame):
    fail_flag_series = df.groupby("component_instance_id")["component_failure"].any()
    return fail_flag_series


def create_train_calibration_test_dfs(
    df: pl.DataFrame, train_stop_date: datetime, calibration_stop_date: datetime
):
    train_df = df.filter(pl.col("date") < train_stop_date)
    calibration_df = df.filter(
        pl.col("date").is_between(train_stop_date, calibration_stop_date, closed="left")
    )
    test_df = df.filter(pl.col("date") > calibration_stop_date)
    return train_df, calibration_df, test_df


def round_to_nearest_05_array(arr: np.ndarray):
    return np.round(arr * 20) / 20


def event_pivoter(df: pl.DataFrame, col: str):
    """will create a pivoted dataset. Used in PM project to
    create 1 record per machineid per day and track various events
    on that day i.e. component 4 failure = true

    Args:
        df (pl.DataFrame): long form df
        col (str): column event

    Returns:
        _type_: pivoted df
    """
    return (
        df.with_columns(
            [
                pl.col("datetime").dt.date().alias("date"),
                pl.lit(1).alias(f"{col}_event"),
            ]
        )
        .pivot(
            values=f"{col}_event",
            index=["date", "machineID"],
            columns=col,
            aggregate_function="first",
            sort_columns=True,
        )
        .fill_null(0)
    )


def create_train_test_df(
    df: pl.DataFrame, split_date: datetime, shift_telem_days: int = 2
):
    """
    This method splits the dataset into a train test split.
    Because this is a predictive maintenace application, the training data shifts the
    telemetry column forward two days to align with the needs of the use case. i.e. predict machine
    failure with telemetry data from previous days. In the test dataset the telemetry data is not shifted
    because we are looking for the signal to provide an early warning of machine failure.
    """
    telemetry_columns = [
        "mean_daily_voltage",
        "mean_daily_rotation",
        "mean_daily_pressure",
        "mean_daily_vibration",
    ]
    train_df = (
        df.filter(pl.col("date") < split_date)
        .sort(["component_instance_id", "date"])
        .with_columns(
            [
                pl.col(telemetry_col)
                .shift(shift_telem_days)
                .over("component_instance_id")
                .fill_null(strategy="mean")
                for telemetry_col in telemetry_columns
            ]
        )
    )

    test_df = df.filter(pl.col("date") > split_date)
    return train_df, test_df


def create_time_to_fail_df(df_fail: pl.DataFrame, window_cols):
    sort_cols = window_cols + ["datetime"]
    time_to_fail_df = (
        df_fail.sort(sort_cols)
        .with_columns(
            [pl.col("datetime").shift(-1).over(window_cols).alias("next_failure_time")]
        )
        .with_columns(
            [
                (pl.col("next_failure_time") - pl.col("datetime"))
                .dt.total_days()
                .alias("time_between_failures")
            ]
        )
        .drop_nulls()
    )
    return time_to_fail_df


def create_fail_plot_df(df_fail: pl.DataFrame, fail_col: str) -> pl.DataFrame:
    return (
        df_fail.select(pl.col(fail_col)).to_series().value_counts().sort(fail_col)
    ).with_columns(percent=pl.col("count") / pl.col("count").sum() * 100)


def create_main_fail_flag_df(df_main, df_fail) -> pl.DataFrame:
    return df_main.join(
        df_fail.with_columns(pl.lit(1).alias("fail_flag")),
        left_on=df_main.columns,
        right_on=df_fail.columns,
        how="left",
    )


def create_anomaly_df(
    df: pl.DataFrame,
    mean_telemetry_cols: List[str],
    window_size: int = 21,
    z_anomaly_threshold: float = 2.3,
) -> pl.DataFrame:
    return (
        df.sort(["machineID", "date"])
        .group_by("machineID", maintain_order=True)
        .map_groups(
            lambda machine: (
                machine
                # Add Rolling Mean and STD
                .with_columns(
                    [
                        pl.col(col)
                        .rolling_mean(window_size=f"{window_size}d", by="date")
                        .alias(f"rolling_avg_{window_size}d_{col}")
                        for col in mean_telemetry_cols
                    ]
                    + [
                        pl.col(col)
                        .rolling_std(window_size=f"{window_size}d", by="date")
                        .fill_null(strategy="backward")
                        .alias(f"rolling_std_{window_size}d_{col}")
                        for col in mean_telemetry_cols
                    ]
                )
                # Add z-scores using newly created columns
                .with_columns(
                    [
                        (
                            (pl.col(col) - pl.col(f"rolling_avg_{window_size}d_{col}"))
                            / pl.col(f"rolling_std_{window_size}d_{col}")
                        ).alias(f"zscore_{window_size}d_{col}")
                        for col in mean_telemetry_cols
                    ]
                ).with_columns(
                    [
                        pl.when(
                            pl.col(f"zscore_{window_size}d_{col}").abs()
                            >= z_anomaly_threshold
                        )
                        .then(1)
                        .otherwise(0)
                        .alias(f"anomaly_flag_{window_size}d_{col}")
                        for col in mean_telemetry_cols
                    ]
                )
            )
        )
    )


def create_main_prejoin_df(
    df: pl.DataFrame, start_date: datetime, end_date: datetime
) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("datetime").dt.date().alias("date")
        )  # Create Date Column
        .drop("datetime")  # ditch datetime
        .sort(["machineID", "comp", "date"])  # sort for instance count
        .with_columns(
            [
                pl.cum_count()
                .over(["machineID", "comp"])
                .alias("component_instance")
                .cast(pl.Utf8),  # instance count
                pl.col("date")
                .shift(-1)
                .over(["machineID", "comp"])
                .alias("end_date"),  # create end_date
            ]
        )
        .with_columns(
            [
                (
                    pl.col("machineID").cast(pl.Utf8)
                    + pl.lit("-")
                    + pl.col("comp")
                    + pl.lit("-")
                    + pl.col("component_instance")
                ).alias(
                    "component_instance_id"
                ),  # create component instance id
                pl.col("end_date")
                .fill_null(pl.lit(end_date))
                .alias("end_date"),  # Fill Nulls on End Date
            ]
        )
        .with_columns(
            [
                pl.date_ranges(
                    start=pl.col("date"),
                    end=pl.col("end_date"),
                    interval="1d",
                    closed="right",
                ).alias("date_range")
            ]  # Create date ranges for component_id
        )
        .explode("date_range")  # Explode
        .drop(["date", "end_date"])  # drop columns
        .rename({"date_range": "date"})  # recreate date column
    )


def create_fail_prejoin_df(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.lit(True).alias("component_failure"),
            pl.col("datetime").dt.date().alias("date"),
            pl.col("failure").alias("comp"),
        ]
    ).drop(["datetime", "failure"])


def create_daily_component_fail_df(
    main_prejoin_df: pl.DataFrame, fail_prejoin_df: pl.DataFrame
) -> pl.DataFrame:
    return (
        main_prejoin_df.join(
            fail_prejoin_df, on=["machineID", "comp", "date"], how="left"
        )
        .with_columns(
            [
                pl.col("component_failure").fill_null(False),
                (pl.col("date") - pl.col("date").min().over("component_instance_id"))
                .dt.days()
                .alias("start"),
            ]
        )
        .with_columns((pl.col("start") + 1).alias("end"))
    )
