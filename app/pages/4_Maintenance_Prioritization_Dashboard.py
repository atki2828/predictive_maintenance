from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots
from utils.ui_components_mpd import (
    render_component_cards,
    render_error_count_display,
    render_input_section,
    render_machine_table,
)

from pmhelpers.dataprocessing import load_data
from pmhelpers.plots import plot_timeseries_stacked_plotly

DASH_DATA_PATH = "./data/dash_demo.csv"
total_dash_df = load_data(DASH_DATA_PATH)
# Set min and max date for date selector
min_date = total_dash_df.select(pl.col("date").min()).item()
max_date = total_dash_df.select(pl.col("date").max()).item() - timedelta(days=1)

st.set_page_config(layout="wide")
st.title("Machine Maintenance Prioritization Dashboard")
st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)
machine_selection = None
selected_date, top_n, sort_option = render_input_section(
    min_date=min_date, max_date=max_date
)
st.divider()

if selected_date and top_n:
    machine_analysis_df = None
    machine_selection, display_df = render_machine_table(
        total_dash_df, selected_date, top_n, sort_option
    )
    st.divider()
    machine_list = machine_selection["selection"].get("rows", None)
    if machine_selection and machine_list:
        # Set machine id
        machine_id = display_df.iloc[machine_list[0]]["machineID"]
        # Create Machine Analysis Df
        machine_analysis_df = total_dash_df.filter(
            (pl.col("machineID") == machine_id) & (pl.col("date") <= selected_date)
        )
        render_error_count_display(
            machine_analysis_df=machine_analysis_df,
            selected_date=selected_date,
            machine_id=machine_id,
        )
        st.divider()
        comp_plot = render_component_cards(machine_analysis_df=machine_analysis_df)

        comp_sensor_lookup = {
            "comp1": ["mean_daily_voltage", "comp1_failure_proba"],
            "comp2": ["mean_daily_rotation", "comp2_failure_proba"],
            "comp3": ["mean_daily_pressure", "comp3_failure_proba"],
            "comp4": ["mean_daily_vibration", "comp4_failure_proba"],
        }

        if comp_plot is not None:
            sensors = comp_sensor_lookup.get(comp_plot)
            plot_fig = plot_timeseries_stacked_plotly(
                machine_analysis_df, sensors=sensors
            )
            st.plotly_chart(plot_fig)
else:
    st.warning("⚠️ Please select a **Date** and **Top N Machines** to begin.")
