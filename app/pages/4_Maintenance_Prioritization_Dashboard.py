from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from pmhelpers.dataprocessing import load_data

DASH_DATA_PATH = "./data/dash_demo.csv"
total_dash_df = load_data(DASH_DATA_PATH)
min_date = total_dash_df.select(pl.col("date").min()).item()
max_date = total_dash_df.select(pl.col("date").max()).item()

st.set_page_config(layout="wide")
st.title("Machine Maintenance Prioritization Dashboard")

st.subheader("Step 1: Input Parameters")
col1, col2, col3 = st.columns(3)

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

st.markdown("---")

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
    if sort_option == "Max Component Probability":
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
    st.subheader(f"Step 2: Top {top_n} Machines by {sort_option}")

    # Column configs: Progress bar for failure probs, TextColumn for machineID
    column_config = {
        "machineID": st.column_config.TextColumn(
            "Machine ID", width="small", help="Unique Machine Identifier"
        )
    }
    # Add progress bars for failure probability columns
    column_config.update(
        {
            col: st.column_config.ProgressColumn(
                col, format="%.2f%%", min_value=0.0, max_value=1.0
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

    st.markdown("---")

#     # --- SECTION 3: Dropdown Selections ---
#     st.subheader("Step 3: Select Machine and Component")
#     sel_col1, sel_col2 = st.columns(2)

#     with sel_col1:
#         selected_machine = st.selectbox(
#             "Select Machine ID:", options=[""] + top_df["Machine ID"].tolist()
#         )

#     with sel_col2:
#         selected_component = st.selectbox(
#             "Select Component:", options=[""] + components
#         )

#     st.markdown("---")

#     # --- SECTION 4: Drill-Down ---
#     if selected_machine and selected_component:
#         st.subheader("Step 4: Drill-Down Details")
#         prob = df.loc[df["Machine ID"] == selected_machine, selected_component].values[
#             0
#         ]

#         # Metrics Row
#         metric_col1, metric_col2, metric_col3 = st.columns(3)
#         with metric_col1:
#             st.metric("Current Status", "Running")
#         with metric_col2:
#             st.metric("Mean Time to Failure", f"{np.random.randint(10,50)} days")
#         with metric_col3:
#             st.metric("Last Maintenance", str(date.today()))

#         # Chart for all component probabilities
#         st.write("#### Component Probabilities for Selected Machine")
#         component_probs = df.loc[df["Machine ID"] == selected_machine, components].melt(
#             var_name="Component", value_name="Failure Probability"
#         )
#         st.bar_chart(component_probs.set_index("Component"))

#     else:
#         st.info("Please select both Machine ID and Component to view details.")

else:
    st.warning("Please select a date and enter a valid Top N number to proceed.")
