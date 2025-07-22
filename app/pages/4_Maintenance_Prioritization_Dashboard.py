from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

from pmhelpers.dataprocessing import load_data

# from pmhelpers.plots import plot_timeseries_stacked_plotly


def plot_timeseries_stacked_plotly(
    df: pd.DataFrame, sensors: list, time_col="date", machine_id=None
):
    # Convert Polars to Pandas if needed
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Filter by machine ID if provided
    if machine_id is not None and "machineID" in df.columns:
        df = df[df["machineID"] == machine_id]

    # Define event styles
    event_styles = {
        "error": {
            "columns": ["error1", "error2", "error3", "error4", "error5"],
            "colors": ["orange", "blueviolet", "brown", "firebrick", "violet"],
            "dash": "dot",
        },
        "failure": {
            "columns": [
                "comp1_failure",
                "comp2_failure",
                "comp3_failure",
                "comp4_failure",
            ],
            "colors": ["red", "green", "black", "blue"],
            "dash": "solid",
        },
        "maintenance": {
            "columns": ["comp1", "comp2", "comp3", "comp4"],
            "colors": ["red", "green", "black", "blue"],
            "dash": "dash",
        },
    }

    # Create stacked subplots
    n = len(sensors)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[s.replace("_", " ").title() for s in sensors],
    )

    # Add sensor time series
    for i, sensor in enumerate(sensors, start=1):
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[sensor],
                mode="lines",
                name=sensor.replace("_", " ").title(),
                line=dict(color="#1f77b4", width=2),
                showlegend=False,  # legend for events only
            ),
            row=i,
            col=1,
        )

    # Build legend items (one copy per event type/color)
    legend_items = []
    for event_type in ["failure", "maintenance", "error"]:
        style = event_styles[event_type]
        for col, color in zip(style["columns"], style["colors"]):
            legend_items.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=color, dash=style["dash"], width=2),
                    name=col.replace("_", " "),
                )
            )

    # Add all legend traces
    for item in legend_items:
        fig.add_trace(item, row=1, col=1)

    # Add vertical event lines per subplot
    for i, sensor in enumerate(sensors, start=1):
        yref = f"y{i}" if i > 1 else "y"
        for event_type in ["failure", "maintenance", "error"]:
            style = event_styles[event_type]
            for col, color in zip(style["columns"], style["colors"]):
                event_times = df.loc[df[col] == 1, time_col]
                for t in event_times:
                    fig.add_shape(
                        type="line",
                        x0=t,
                        x1=t,
                        y0=df[sensor].min(),
                        y1=df[sensor].max(),
                        xref="x",
                        yref=yref,
                        line=dict(color=color, dash=style["dash"], width=1.5),
                    )

    # Layout for Streamlit look
    fig.update_layout(
        height=260 * n,
        title="Telemetry Time Series",
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Event Legend",
        ),
        margin=dict(l=40, r=200, t=60, b=40),
    )

    # Update axis labels
    fig.update_xaxes(title_text="Date", row=n, col=1)
    fig.update_yaxes(title_text="", showgrid=True)

    return fig


DASH_DATA_PATH = "./data/dash_demo.csv"
total_dash_df = load_data(DASH_DATA_PATH)
min_date = total_dash_df.select(pl.col("date").min()).item()
max_date = total_dash_df.select(pl.col("date").max()).item()

st.set_page_config(layout="wide")
st.title("Machine Maintenance Prioritization Dashboard")

st.subheader("Input Parameters")
col1, col2, col3 = st.columns(3)

machine_selection = None
with col1:
    selected_date = st.date_input(
        "Select Date:", value=None, min_value=min_date, max_value=max_date
    )

with col2:
    top_n = st.number_input("Top N Machines:", min_value=1, max_value=20, value=10)

with col3:
    sort_option = st.selectbox(
        "Sort By:",
        options=[
            "Max Component Failure Probability",
            "Average of Component Failure Probabilities",
        ],
        index=0,
    )

st.divider()

if selected_date and top_n:
    # --- Setup Display DF ---
    dash_df = total_dash_df.filter(pl.col("date") == selected_date)

    fail_prob_cols = [col for col in dash_df.columns if "proba" in col]
    display_cols = ["machineID"] + fail_prob_cols

    # Rename mapping for fail prob columns
    rename_map = {
        col: f"Comp{i} Failure Probability"
        for i, col in enumerate(fail_prob_cols, start=1)
    }

    # Add SortValue for sorting
    sort_col = "SortValue"
    if sort_option == "Max Component Failure Probability":
        dash_df = dash_df.with_columns(
            pl.max_horizontal([pl.col(c) for c in fail_prob_cols]).alias(sort_col)
        )
    else:
        dash_df = dash_df.with_columns(
            pl.mean_horizontal([pl.col(c) for c in fail_prob_cols]).alias(sort_col)
        )

    # Sort and select top N
    df = dash_df.sort(by=sort_col, descending=True)
    top_df = df.head(top_n)

    # Convert to Pandas for styling
    display_df = top_df.select(display_cols).rename(rename_map).to_pandas()

    # Center all values (including headers)
    styled_df = display_df.style.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ]
    )

    # --- SECTION 2: Show Table ---
    st.subheader(f"Top {top_n} Machines by 2 Day Ahead {sort_option}")
    st.markdown("""**Select Machine ID For Deeper Dive**""")

    # Column configs: Progress bar for failure probs, TextColumn for machineID
    column_config = {
        "machineID": st.column_config.TextColumn(
            "Machine ID",
            width="small",
            help="Unique Machine Identifier",
        )
    }
    # Add progress bars for failure probability columns
    column_config.update(
        {
            col: st.column_config.ProgressColumn(
                col, format="percent", min_value=0.0, max_value=1.0
            )
            for col in rename_map.values()
        }
    )

    machine_selection = st.dataframe(
        styled_df,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode=["single-row"],
        use_container_width=True,
    )

    st.divider()

    print(machine_selection)

    ### use this machine_id for a lookup
    machine_id = display_df.iloc[machine_selection["selection"]["rows"][0]]["machineID"]
    print(machine_id)
    machine_analysis_df = total_dash_df.filter(
        (pl.col("machineID") == machine_id) & (pl.col("date") <= selected_date)
    )

    st.markdown("## 7 Day Error Counts ")
    st.write(f"Machine ID = {int(machine_id)}")

    print(machine_selection)

    print(machine_analysis_df.columns)

    (
        error_col1,
        error_col2,
        error_col3,
        error_col4,
        error_col5,
    ) = st.columns(5)

    with error_col1:
        error_count = (
            machine_analysis_df.filter(
                pl.col("date").is_between(
                    selected_date - timedelta(days=7), selected_date
                )
            )
            .select("error1")
            .sum()
            .item()
        )
        print(error_count)
        st.metric(label="Error 1", value=error_count, border=True)
    with error_col2:
        error_count = (
            machine_analysis_df.filter(
                pl.col("date").is_between(
                    selected_date - timedelta(days=7), selected_date
                )
            )
            .select("error2")
            .sum()
        )
        st.metric(label="Error 2", value=error_count, border=True)
    with error_col3:
        error_count = (
            machine_analysis_df.filter(
                pl.col("date").is_between(
                    selected_date - timedelta(days=7), selected_date
                )
            )
            .select("error3")
            .sum()
        )
        st.metric(label="Error 3", value=error_count, border=True)
    with error_col4:
        error_count = (
            machine_analysis_df.filter(
                pl.col("date").is_between(
                    selected_date - timedelta(days=7), selected_date
                )
            )
            .select("error4")
            .sum()
        )
        st.metric(label="Error 4", value=error_count, border=True)
    with error_col5:
        error_count = (
            machine_analysis_df.filter(
                pl.col("date").is_between(
                    selected_date - timedelta(days=7), selected_date
                )
            )
            .select("error5")
            .sum()
        )
        st.metric(label="Error 5", value=error_count, border=True)

    st.divider()
# Example selected row
if machine_selection is not None:
    comp_plot = None
    row = machine_analysis_df.to_pandas().iloc[-1]

    # Components
    components = ["comp1", "comp2", "comp3", "comp4"]

    # Top-level columns for each component
    top_cols = st.columns(4)

    for i, comp in enumerate(components):
        with top_cols[i]:
            clicked = st.button(label=comp.capitalize(), icon="🚨")
            try:
                if clicked:
                    comp_plot = comp
            except Exception as e:
                print(e)

            failure_prob = row[f"{comp}_failure_proba"]
            install_date_raw = row[f"component_install_date_{comp}"]
            days_running = row[f"end_{comp}"]

            # Format date as Month Day, Year
            if isinstance(install_date_raw, str):
                try:
                    install_date = datetime.strptime(
                        install_date_raw, "%Y-%m-%d"
                    ).strftime("%b %d, %Y")
                except:
                    install_date = install_date_raw
            elif isinstance(install_date_raw, datetime):
                install_date = install_date_raw.strftime("%b %d, %Y")
            else:
                install_date = str(install_date_raw)

            # Display stacked metrics (centered naturally by Streamlit)
            st.metric(
                label="2 Day Failure Probability",
                value=f"{failure_prob:.2%}",
                border=True,
            )
            st.metric(label="Install Date", value=install_date, border=True)
            st.metric(label="Days Running", value=int(days_running), border=True)
            #### Plotting Code Check
    sensors = [
        "mean_daily_vibration",
        "comp4_failure_proba",
    ]
    comp_sensor_lookup = {
        "comp1": ["mean_daily_voltage", "comp1_failure_proba"],
        "comp2": ["mean_daily_rotation", "comp2_failure_proba"],
        "comp3": ["mean_daily_pressure", "comp3_failure_proba"],
        "comp4": ["mean_daily_vibration", "comp4_failure_proba"],
    }

    if comp_plot is not None:
        sensors = comp_sensor_lookup.get(comp_plot)
        plot_fig = plot_timeseries_stacked_plotly(machine_analysis_df, sensors=sensors)
        st.plotly_chart(plot_fig)
