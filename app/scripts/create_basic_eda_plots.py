from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st
from lifelines import WeibullFitter

from pmhelpers.dataprocessing import create_fail_plot_df
from pmhelpers.models import create_survival_model_dict
from pmhelpers.plots import (
    plot_box_and_strip,
    plot_failure_counts,
    plot_time_between_failures_dist,
    plot_timeseries_stacked,
    survival_hazard_group_plotter,
)

plt.style.use("ggplot")
TELEMETRY_FILE_PATH = "./app/data/PdM_telemetry.csv"
FAILURE_FILE_PATH = "./app/data/PdM_failures.csv"
MACHINE_FILE_PATH = "./app/data/PdM_machines.csv"
MAINTENANCE_FILE_PATH = "./app/data/PdM_maintenance.csv"
ERROR_FILE_PATH = "./app/data/PdM_errors.csv"

DAILY_PLOT_DATA_PATH = "./app/data/daily_plot_df.csv"

COMP_FAIL_PLOT_PATH = "./app/data/comp_fail_plot.csv"
MACHINE_FAIL_PLOT_PATH = "./app/data/machine_fail_plot.csv"
TIME_BETWEEN_FAIL_FLAG_PATH = "./app/data/time_between_fail_flag.csv"
TIME_TO_FAIL_PATH = "./app/data/time_to_fail.csv"


def create_plot_df(
    df: pl.DataFrame, machine_id: int, start_date: datetime
) -> pl.DataFrame:
    # Fixed interval for plotting = 30 days
    end_date = start_date + timedelta(days=30)
    return (
        df.filter(pl.col("machineID") == machine_id)
        .filter(pl.col("date").is_between(start_date, end_date))
        .sort("date", descending=False)
    )


def main():
    comp_fail_plot_df = pl.read_csv(COMP_FAIL_PLOT_PATH, try_parse_dates=True)
    time_to_fail_df = pl.read_csv(TIME_TO_FAIL_PATH, try_parse_dates=True)
    time_between_comp_fail_flag_df = pl.read_csv(TIME_BETWEEN_FAIL_FLAG_PATH)
    component_weibull_dict = create_survival_model_dict(
        fail_flag_df=time_between_comp_fail_flag_df,
        model=WeibullFitter,
        group_level_col="model",
        duration_col="time_between_maintenance",
        observe_col="fail_flag",
    )

    failure_count_fig = plot_failure_counts(comp_fail_plot_df)
    failure_count_fig.savefig(
        "app/static_plots/failure_count.png", dpi=300, bbox_inches="tight"
    )
    time_between_fail_dist_fig = plot_time_between_failures_dist(time_to_fail_df)
    time_between_fail_dist_fig.savefig(
        "app/static_plots/time_between_comp_fail_dist.png", dpi=300, bbox_inches="tight"
    )

    time_to_fail_by_comp_fig = plot_box_and_strip(
        df=time_to_fail_df.sort("failure").to_pandas(),
        x="failure",
        y="time_between_failures",
        title="Time Between Failures By Component",
    )
    time_to_fail_by_comp_fig.savefig(
        "app/static_plots/time_to_fail_by_comp.png", dpi=300, bbox_inches="tight"
    )
    compare_fig = plot_box_and_strip(
        time_between_comp_fail_flag_df.to_pandas(),
        x="fail_flag",
        y="time_between_maintenance",
        title="Time Between Maintenance and Failures",
    )

    compare_fig.savefig(
        "app/static_plots/compare_fig.png", dpi=300, bbox_inches="tight"
    )

    survival_fig = survival_hazard_group_plotter(
        component_weibull_dict, model_name="Weibull Model"
    )
    survival_fig.savefig("app/static_plots/survival.png", dpi=300, bbox_inches="tight")
    # Load data
    comp_fail_plot_daily_df = pl.read_csv(DAILY_PLOT_DATA_PATH, try_parse_dates=True)
    # Set mean sensor cols
    sensor_cols = [
        col for col in comp_fail_plot_daily_df.columns if col.startswith("mean")
    ]

    ### Feature Eng Telemetry Plots
    # Set Dates
    start_date_1 = datetime(2015, 3, 10)
    start_date_2 = datetime(2015, 6, 20)
    start_date_3 = datetime(2015, 5, 25)
    start_date_4 = datetime(2015, 10, 31)

    # Create comp dfs
    comp1_failure_df = create_plot_df(
        df=comp_fail_plot_daily_df, machine_id=79, start_date=start_date_1
    )
    comp2_failure_df = create_plot_df(
        df=comp_fail_plot_daily_df, machine_id=23, start_date=start_date_2
    )
    comp3_failure_df = create_plot_df(
        df=comp_fail_plot_daily_df, machine_id=42, start_date=start_date_3
    )
    comp4_failure_df = create_plot_df(
        df=comp_fail_plot_daily_df,
        machine_id=51,
        start_date=start_date_4,
    )

    # Create Figs
    comp1_fig = plot_timeseries_stacked(
        comp1_failure_df.to_pandas(), sensors=sensor_cols
    )
    comp2_fig = plot_timeseries_stacked(
        comp2_failure_df.to_pandas(), sensors=sensor_cols
    )
    comp3_fig = plot_timeseries_stacked(
        comp3_failure_df.to_pandas(), sensors=sensor_cols
    )
    comp4_fig = plot_timeseries_stacked(
        comp4_failure_df.to_pandas(), sensors=sensor_cols
    )
    comp1_fig.savefig("app/static_plots/comp1_fig.png", dpi=300, bbox_inches="tight")
    comp2_fig.savefig("app/static_plots/comp2_fig.png", dpi=300, bbox_inches="tight")
    comp3_fig.savefig("app/static_plots/comp3_fig.png", dpi=300, bbox_inches="tight")
    comp4_fig.savefig("app/static_plots/comp4_fig.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
