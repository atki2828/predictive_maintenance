from datetime import date, datetime, timedelta

import pandas as pd
import polars as pl
import streamlit as st


def render_input_section(min_date: date, max_date: date):
    """
    Render the first section of Maintenance_Prioritization_Dashboard

    Returns:
        tuple: (selected_date, top_n, sort_option)
    """
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

    return selected_date, top_n, sort_option


def render_machine_table(
    total_dash_df: pl.DataFrame, selected_date, top_n, sort_option
):
    """
    Render the second section: a table of top machines by failure probability.

    Args:
        total_dash_df (pl.DataFrame): The full dashboard data.
        selected_date (date): Date selected by the user.
        top_n (int): Number of machines to display.
        sort_option (str): Sorting option from UI.

    Returns:
        dict or None: Selection info from Streamlit dataframe.
    """
    # --- Setup Display DF ---
    dash_df = total_dash_df.filter(pl.col("date") == selected_date)
    fail_prob_cols = [col for col in dash_df.columns if "proba" in col]
    display_cols = ["machineID"] + fail_prob_cols

    # Rename columns for display
    rename_map = {
        col: f"Comp{i} Failure Probability"
        for i, col in enumerate(fail_prob_cols, start=1)
    }

    # Add sort value for ranking
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

    # Convert to pandas for Streamlit styling
    display_df = top_df.select(display_cols).rename(rename_map).to_pandas()

    # Style: center all values
    styled_df = display_df.style.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ]
    )

    st.subheader(f"Top {top_n} Machines by 2 Day Ahead {sort_option}")
    st.markdown("**Select Machine ID For Deeper Dive**")

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

    # Render the interactive table
    selection = st.dataframe(
        styled_df,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode=["single-row"],
        use_container_width=True,
    )

    return selection, display_df


def render_error_count_display(
    machine_analysis_df: pl.DataFrame, selected_date: datetime, machine_id: int
):
    st.markdown("## 7 Day Error Counts ")
    st.write(f"Machine ID = {int(machine_id)}")
    error_cols = st.columns(5)

    # Error labels
    error_labels = [f"Error {i}" for i in range(1, 6)]
    error_fields = [f"error{i}" for i in range(1, 6)]

    # Calculate date range
    date_start = selected_date - timedelta(days=7)
    date_end = selected_date

    # Loop through columns and display metrics
    for col, error_field, label in zip(error_cols, error_fields, error_labels):
        with col:
            error_count = (
                machine_analysis_df.filter(
                    pl.col("date").is_between(date_start, date_end)
                )
                .select(error_field)
                .sum()
                .item()
            )
            st.metric(label=label, value=error_count, border=True)


def render_component_cards(
    machine_analysis_df, components: list = ["comp1", "comp2", "comp3", "comp4"]
) -> str:
    """
    Render component cards with metrics and return the clicked component.

    Args:
        machine_analysis_df (pd.DataFrame or Polars DataFrame): The analysis data.
        components (list): List of components like ["comp1", "comp2", "comp3", "comp4"].

    Returns:
        str or None: The component clicked by the user, or None if nothing selected.
    """
    # Convert to pandas if Polars
    if hasattr(machine_analysis_df, "to_pandas"):
        df_pd = machine_analysis_df.to_pandas()
    else:
        df_pd = machine_analysis_df

    # Take last row for details
    row = df_pd.iloc[-1]

    cols = st.columns(len(components))
    clicked_component = None

    for i, comp in enumerate(components):
        with cols[i]:
            st.markdown(f"### {comp.capitalize()}")

            # Extract data
            failure_prob = row.get(f"{comp}_failure_proba")
            install_date_raw = row.get(f"component_install_date_{comp}")
            days_running = row.get(f"end_{comp}")

            # Format date
            if isinstance(install_date_raw, str):
                try:
                    install_date = datetime.strptime(
                        install_date_raw, "%Y-%m-%d"
                    ).strftime("%b %d, %Y")
                except ValueError:
                    install_date = install_date_raw
            elif isinstance(install_date_raw, datetime):
                install_date = install_date_raw.strftime("%b %d, %Y")
            else:
                install_date = str(install_date_raw) if install_date_raw else "N/A"

            # Show metrics
            st.metric(
                "2 Day Failure Probability",
                f"{failure_prob:.2%}" if failure_prob else "N/A",
                border=True,
            )
            st.metric("Install Date", install_date, border=True)
            st.metric(
                "Days Running",
                int(days_running) if days_running else "N/A",
                border=True,
            )
            st.divider()

            # Button
            if st.button(label=comp.capitalize(), icon="🚨"):
                clicked_component = comp

    return clicked_component
