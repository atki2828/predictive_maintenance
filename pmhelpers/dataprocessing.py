from datetime import datetime
from functools import reduce
from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st


@st.cache_data
def load_data(file_path: str) -> pl.DataFrame:
    """
    Load a CSV file into a Polars DataFrame with date parsing enabled.

    This function uses Streamlit's caching mechanism to avoid reloading
    the data on every rerun of the app, improving performance.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to load.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the loaded data with parsed dates.
    """
    df: pl.DataFrame = pl.read_csv(file_path, try_parse_dates=True)
    return df


def create_time_between_fail_group_df(
    fail_flag_df: pl.DataFrame, window_cols: List[str], max_datetime: datetime
) -> pl.DataFrame:
    """
    Compute the time between consecutive maintenance or failure events for each group.

    This function calculates the time in days between a maintenance event and
    the next event (maintenance or failure) within a given grouping window.
    If there is no next event, the time difference is computed using the
    provided `max_datetime`.

    Parameters
    ----------
    fail_flag_df : pl.DataFrame
        Input DataFrame containing at least 'datetime' and 'fail_flag' columns.
    window_cols : list[str]
        List of column names to define grouping (e.g., machineID, component).
    max_datetime : datetime
        Maximum datetime used to fill null values when there is no next event.

    Returns
    -------
    pl.DataFrame
        DataFrame with additional columns:
        - `next_replace_time`: Timestamp of the next event in the group.
        - `time_between_maintenance`: Time difference in days between events.
    """
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


def get_failures_by_comp(
    df_fail: pl.DataFrame, machine_id: int, comp: str
) -> pl.DataFrame:
    """
    Filter failure records for a specific machine and component.

    Parameters
    ----------
    df_fail : pl.DataFrame
        Polars DataFrame containing failure records with 'machineID' and 'comp' columns.
    machine_id : int
        Machine ID to filter on.
    comp : str
        Component name to filter on.

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame containing only the specified machine and component.
    """
    return df_fail.filter(pl.col("machineID") == machine_id).filter(
        pl.col("comp") == comp
    )


def get_component_failure(df: pd.DataFrame) -> pd.Series:
    """
    Determine if each component instance experienced a failure.

    Groups data by 'component_instance_id' and returns a boolean Series
    indicating whether each component instance failed at least once.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing 'component_instance_id' and 'component_failure' columns.

    Returns
    -------
    pd.Series
        Boolean Series indexed by 'component_instance_id', where True indicates failure.
    """
    fail_flag_series = df.groupby("component_instance_id")["component_failure"].any()
    return fail_flag_series


def create_train_calibration_test_dfs(
    df: pl.DataFrame, train_stop_date: datetime, calibration_stop_date: datetime
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a DataFrame into training, calibration, and test sets based on date ranges.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing a 'date' column.
    train_stop_date : datetime
        End date for the training set (exclusive).
    calibration_stop_date : datetime
        End date for the calibration set (exclusive).

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (train_df, calibration_df, test_df) split by date ranges:
        - train_df: dates < train_stop_date
        - calibration_df: train_stop_date <= dates < calibration_stop_date
        - test_df: dates > calibration_stop_date
    """
    train_df = df.filter(pl.col("date") < train_stop_date)
    calibration_df = df.filter(
        pl.col("date").is_between(train_stop_date, calibration_stop_date, closed="left")
    )
    test_df = df.filter(pl.col("date") > calibration_stop_date)
    return train_df, calibration_df, test_df


def round_to_nearest_05_array(arr: np.ndarray) -> np.ndarray:
    """
    Round the elements of an array to the nearest 0.05.

    Each value is scaled by 20, rounded to the nearest integer, and then
    scaled back to achieve rounding to the nearest 0.05.

    Parameters
    ----------
    arr : np.ndarray
        Input array of numeric values.

    Returns
    -------
    np.ndarray
        Array with values rounded to the nearest 0.05.

    Examples
    --------
    >>> import numpy as np
    >>> round_to_nearest_05_array(np.array([1.02, 2.13, 3.27]))
    array([1.  , 2.15, 3.25])
    """
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
        "anomaly_flag_21d_mean_daily_voltage",
        "anomaly_flag_21d_mean_daily_rotation",
        "anomaly_flag_21d_mean_daily_pressure",
        "anomaly_flag_21d_mean_daily_vibration",
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


def create_time_to_fail_df(
    df_fail: pl.DataFrame, window_cols: List[str]
) -> pl.DataFrame:
    """
    Create a DataFrame showing time between consecutive failures within specified groups.

    This function sorts the data by `window_cols` and 'datetime', then calculates
    the time in days to the next failure event for each row.

    Parameters
    ----------
    df_fail : pl.DataFrame
        Polars DataFrame containing failure events with a 'datetime' column.
    window_cols : list of str
        Column names to group by (e.g., machineID, component).

    Returns
    -------
    pl.DataFrame
        DataFrame with additional columns:
        - 'next_failure_time': Timestamp of the next failure within the group.
        - 'time_between_failures': Time difference in days between consecutive failures.

    Examples
    --------
    >>> create_time_to_fail_df(df_fail, ["machineID", "comp"])
    """
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
    """
    Aggregate failure counts and compute percentages for plotting.

    Parameters
    ----------
    df_fail : pl.DataFrame
        Polars DataFrame containing failure records.
    fail_col : str
        Column name representing the failure category (e.g., component).

    Returns
    -------
    pl.DataFrame
        DataFrame with:
        - 'fail_col': Unique failure categories.
        - 'count': Frequency of each category.
        - 'percent': Percentage share of each category.

    Examples
    --------
    >>> create_fail_plot_df(df_fail, "comp")
    """
    return (
        df_fail.select(pl.col(fail_col)).to_series().value_counts().sort(fail_col)
    ).with_columns(percent=pl.col("count") / pl.col("count").sum() * 100)


def create_main_fail_flag_df(
    df_main: pl.DataFrame, df_fail: pl.DataFrame
) -> pl.DataFrame:
    """
    Merge main dataset with failure records and add a failure flag.

    Parameters
    ----------
    df_main : pl.DataFrame
        Main Polars DataFrame (e.g., all observations).
    df_fail : pl.DataFrame
        Polars DataFrame containing failure events with matching columns.

    Returns
    -------
    pl.DataFrame
        Merged DataFrame with an additional column:
        - 'fail_flag': 1 if a matching failure exists, else null.

    Examples
    --------
    >>> create_main_fail_flag_df(df_main, df_fail)
    """
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
    """
    Detect anomalies in telemetry signals using rolling statistics and z-scores.

    For each machine, this function calculates rolling means and standard deviations
    over a given window size for selected telemetry columns, then computes z-scores.
    Observations where the absolute z-score exceeds the threshold are flagged as anomalies.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing telemetry data with 'machineID' and 'date' columns.
    mean_telemetry_cols : list of str
        List of telemetry column names for anomaly detection.
    window_size : int, default=21
        Rolling window size in days.
    z_anomaly_threshold : float, default=2.3
        Threshold for absolute z-score to flag an anomaly.

    Returns
    -------
    pl.DataFrame
        DataFrame with additional columns for:
        - Rolling averages: `rolling_avg_{window_size}d_<col>`
        - Rolling std: `rolling_std_{window_size}d_<col>`
        - Z-scores: `zscore_{window_size}d_<col>`
        - Anomaly flags: `anomaly_flag_{window_size}d_<col>` (1 = anomaly, 0 = normal)

    Examples
    --------
    >>> create_anomaly_df(df, mean_telemetry_cols=["volt", "rotate"], window_size=21, z_anomaly_threshold=2.3)
    """
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
                )
                # Add anomaly flags based on z-score threshold
                .with_columns(
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
    """
    Transform the raw maintenance DataFrame into a pre-join format for modeling.

    This function processes component-level records and creates:
    - A `component_instance_id` for each unique machine-component instance.
    - A continuous daily date range from the start to the end of each component instance.
    - Necessary fields for joining telemetry and failure information later.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame containing at least 'datetime', 'machineID', and 'comp' columns.
    start_date : datetime
        Minimum date for filtering or reference (currently unused, but included for flexibility).
    end_date : datetime
        End date used to fill missing `end_date` values for active components.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with:
        - 'component_instance_id': Unique ID combining machineID, component, and instance number.
        - 'date': Daily timestamps for each component instance (exploded from ranges).
        - All original identifying columns needed for future joins.

    Notes
    -----
    - The function assumes that the input DataFrame contains a `datetime` column.
    - Instances are created based on cumulative count per machine-component group.

    Examples
    --------
    >>> create_main_prejoin_df(df, start_date=datetime(2015,1,1), end_date=datetime(2016,1,1))
    """
    return (
        df.with_columns(
            pl.col("datetime").dt.date().alias("date")
        )  # Convert datetime to date
        .drop("datetime")
        .sort(["machineID", "comp", "date"])  # Sort for consistent cumulative counting
        .with_columns(
            [
                pl.cum_count()
                .over(["machineID", "comp"])
                .alias("component_instance")
                .cast(pl.Utf8),  # Instance number as string
                pl.col("date")
                .shift(-1)
                .over(["machineID", "comp"])
                .alias("end_date"),  # Next component start as end_date
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
                ),  # Unique instance ID
                pl.col("end_date")
                .fill_null(pl.lit(end_date))
                .alias("end_date"),  # Fill null end dates with global end_date
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
            ]  # Create full daily date range for each component instance
        )
        .explode("date_range")  # Expand date ranges into individual rows
        .drop(["date", "end_date"])  # Drop intermediate columns
        .rename({"date_range": "date"})  # Rename exploded column to date
    )


def create_fail_prejoin_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform the raw failure records into a pre-join format.

    This function prepares failure data for merging with the main dataset by:
    - Creating a `component_failure` flag for all records.
    - Converting the datetime column to a date column.
    - Renaming the `failure` column to `comp` for consistency.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing at least 'datetime' and 'failure' columns.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with:
        - 'component_failure': Boolean flag set to True for all rows.
        - 'date': Extracted date from the datetime column.
        - 'comp': Component name copied from the original 'failure' column.

    """
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
    """
    Create a daily-level component DataFrame with failure flags.

    Joins the main pre-join dataset with the failure dataset and computes:
    - `component_failure`: Boolean flag for failures (default False if missing).
    - `start`: Day index within the component instance timeline.
    - `end`: Day index + 1.

    Parameters
    ----------
    main_prejoin_df : pl.DataFrame
        DataFrame with daily component records (from create_main_prejoin_df).
    fail_prejoin_df : pl.DataFrame
        DataFrame with failure events (from create_fail_prejoin_df).

    Returns
    -------
    pl.DataFrame
        Daily-level component data including failure flags and timeline indices.
    """
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


def create_fail_prob_dash_demo_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create a combined failure probability DataFrame for all components.

    Filters and selects failure probabilities for each component, then joins them
    into a single DataFrame keyed by `date` and `machineID`.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing component failure probabilities.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: `date`, `machineID`, and failure probabilities for all components.
    """
    df_list = [
        df.filter(pl.col("comp") == comp)
        .select(["date", "machineID", f"{comp}_failure_proba"])
        .unique()
        for comp in ["comp1", "comp2", "comp3", "comp4"]
    ]
    return reduce(
        lambda left, right: left.join(right, on=["date", "machineID"], how="inner"),
        df_list,
    )


def create_comp_dash_demo_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create a pivoted DataFrame summarizing component details for each machine and date.

    This function aggregates component-level data into a wide format for dashboard use,
    including:
    - Component end dates
    - Component instance identifiers
    - Component installation dates

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with component-level records.

    Returns
    -------
    pl.DataFrame
        Pivoted DataFrame with one row per machine-date and columns for component details.
    """
    result = (
        df.select(
            [
                "machineID",
                "date",
                "comp",
                "component_instance_id",
                "component_instance",
                "start",
                "end",
            ]
        )
        .with_columns(
            pl.col("date")
            .first()
            .over("component_instance_id")
            .alias("component_install_date")
        )
        .pivot(
            index=["machineID", "date"],
            columns="comp",
            values=["end", "component_instance", "component_install_date"],
            aggregate_function="first",
        )
    )

    # Clean up column names for readability
    result.columns = [col.replace("_comp_", "_") for col in result.columns]

    return result


def create_telemetry_dash_demo_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create a telemetry dataset for dashboard visualization.

    Drops component-specific and failure-related columns, removes duplicates,
    and sorts by date for time-series visualization.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing telemetry data.

    Returns
    -------
    pl.DataFrame
        Cleaned DataFrame with telemetry signals, sorted by date.
    """
    return (
        df.drop(
            [
                "comp",
                "component_instance_id",
                "component_instance",
                "start",
                "end",
                "component_failure",
                "comp1_failure_proba",
                "comp2_failure_proba",
                "comp3_failure_proba",
                "comp4_failure_proba",
            ]
        )
        .unique()
        .sort("date", descending=False)
    )
