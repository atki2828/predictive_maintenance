from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st


@st.cache_data
def load_data(file_path: str) -> pl.DataFrame:
    df = pl.read_csv(file_path)
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
