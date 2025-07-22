from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st
from app_text import (
    feature_text,
    telemetry_component_four_text,
    telemetry_component_one_text,
    telemetry_component_three_text,
    telemetry_component_two_text,
    telemetry_eda_intro,
)
from pmhelpers.dataprocessing import load_data
from pmhelpers.plots import plot_timeseries_stacked

plt.style.use("ggplot")

DAILY_PLOT_DATA_PATH = "./app/data/daily_plot_df.csv"


@st.cache_data
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
    st.set_page_config(layout="wide", page_icon="📈")
    # Load data
    comp_fail_plot_daily_df = load_data(DAILY_PLOT_DATA_PATH)
    # Set mean sensor cols
    sensor_cols = [
        col for col in comp_fail_plot_daily_df.columns if col.startswith("mean")
    ]
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

    st.markdown(telemetry_eda_intro)
    st.divider()

    st.markdown(telemetry_component_one_text)
    st.pyplot(comp1_fig)
    st.divider()

    st.markdown(telemetry_component_two_text)
    st.pyplot(comp2_fig)
    st.divider()

    st.markdown(telemetry_component_three_text)
    st.pyplot(comp3_fig)
    st.divider()

    st.markdown(telemetry_component_four_text)
    st.pyplot(comp4_fig)
    st.divider()

    st.markdown(feature_text)


if __name__ == "__main__":
    main()
