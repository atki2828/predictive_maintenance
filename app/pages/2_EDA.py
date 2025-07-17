import matplotlib.pyplot as plt
import polars as pl
import streamlit as st
from app_text import eda_intro

from pmhelpers.dataprocessing import (
    create_time_between_fail_group_df,
    create_time_to_fail_df,
    load_data,
)
from pmhelpers.plots import (
    plot_box_and_strip,
    plot_failure_counts,
    plot_machine_failure_counts,
    plot_time_between_failures_dist,
    plot_time_between_maintenance_dist,
)

plt.style.use("ggplot")
TELEMETRY_FILE_PATH = "./data/PdM_telemetry.csv"
FAILURE_FILE_PATH = "./data/PdM_failures.csv"
MACHINE_FILE_PATH = "./data/PdM_machines.csv"
MAINTENANCE_FILE_PATH = "./data/PdM_maintenance.csv"
ERROR_FILE_PATH = "./data/PdM_errors.csv"


def main():
    # Set Title
    st.title("Exploratory Data Analysis")

    # Read and Cache Data
    df_tel = load_data(TELEMETRY_FILE_PATH)
    df_err = load_data(ERROR_FILE_PATH)
    df_fail = load_data(FAILURE_FILE_PATH)
    df_mach = load_data(MACHINE_FILE_PATH)
    df_main = load_data(MAINTENANCE_FILE_PATH)

    # fail column is comp in other dfs may need to handle
    window_cols = ["machineID", "failure"]
    time_to_fail_df = create_time_to_fail_df(df_fail, window_cols)

    # Eda Intro Markdown
    st.markdown(eda_intro)

    fail_count_fig = plot_failure_counts(df_fail)
    st.pyplot(fail_count_fig)

    machine_fail_fig = plot_machine_failure_counts(df_fail)
    st.pyplot(machine_fail_fig)

    # TODO: Separate time to fail logic
    time_to_failure_dist_fig = plot_time_between_failures_dist(df_fail)
    st.pyplot(time_to_failure_dist_fig)

    time_to_fail_box_strip_fig = plot_box_and_strip(
        time_to_fail_df,
        x="failure",
        y="time_between_failures",
        title="Time Between Failures By Component",
    )
    st.pyplot(time_to_fail_box_strip_fig)

    # sort_cols = ['datetime'] + window_cols
    main_fail_flag_df = df_main.join(
        df_fail.with_columns(pl.lit(1).alias("fail_flag")),
        left_on=df_main.columns,
        right_on=df_fail.columns,
        how="left",
    )

    window_cols = ["machineID", "comp"]
    max_datetime = df_tel.select(pl.col("datetime").max()).to_series()[0]
    time_between_fail_flag_df = create_time_between_fail_group_df(
        main_fail_flag_df, window_cols, max_datetime
    ).join(df_mach, on="machineID", how="inner")
    time_between_fail_flag_fig = plot_time_between_maintenance_dist(
        time_between_fail_flag_df
    )
    st.pyplot(time_between_fail_flag_fig)

    box_and_strip_fail_fig = plot_box_and_strip(
        time_between_fail_flag_df.to_pandas(),
        x="fail_flag",
        y="time_between_maintenance",
        title="Time Between Maintenance and Failures",
    )
    st.pyplot(box_and_strip_fail_fig)


if __name__ == "__main__":
    main()
