from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("Machine-Component Failure Analysis")

# --- SECTION 1: Inputs ---
st.subheader("Step 1: Input Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    selected_date = st.date_input("Select Date:", value=None)

with col2:
    top_n = st.number_input("Top N Machines:", min_value=1, max_value=10, value=5)

with col3:
    sort_option = st.selectbox(
        "Sort By:",
        options=["Max Component Probability", "Sum of Component Probabilities"],
        index=0,
    )

st.markdown("---")

# Proceed only if date and top_n provided
if selected_date and top_n:
    # --- Generate Dummy Data ---
    np.random.seed(42)
    machines = [f"Machine_{i}" for i in range(1, 11)]
    components = ["Component_A", "Component_B", "Component_C", "Component_D"]

    # Random failure probabilities for each machine-component pair
    data = {"Machine ID": machines}
    for comp in components:
        data[comp] = np.round(np.random.uniform(0.01, 0.5, size=len(machines)), 3)

    df = pd.DataFrame(data)

    # Determine sort key
    if sort_option == "Max Component Probability":
        df["SortValue"] = df[components].max(axis=1)
    else:
        df["SortValue"] = df[components].sum(axis=1)

    # Sort and select top N
    df = df.sort_values(by="SortValue", ascending=False).drop(columns=["SortValue"])
    top_df = df.head(top_n).reset_index(drop=True)

    # --- SECTION 2: Show Table ---
    st.subheader(f"Step 2: Top {top_n} Machines by {sort_option}")
    st.dataframe(top_df, hide_index=True)

    st.markdown("---")

    # --- SECTION 3: Dropdown Selections ---
    st.subheader("Step 3: Select Machine and Component")
    sel_col1, sel_col2 = st.columns(2)

    with sel_col1:
        selected_machine = st.selectbox(
            "Select Machine ID:", options=[""] + top_df["Machine ID"].tolist()
        )

    with sel_col2:
        selected_component = st.selectbox(
            "Select Component:", options=[""] + components
        )

    st.markdown("---")

    # --- SECTION 4: Drill-Down ---
    if selected_machine and selected_component:
        st.subheader("Step 4: Drill-Down Details")
        prob = df.loc[df["Machine ID"] == selected_machine, selected_component].values[
            0
        ]

        # Metrics Row
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Current Status", "Running")
        with metric_col2:
            st.metric("Mean Time to Failure", f"{np.random.randint(10,50)} days")
        with metric_col3:
            st.metric("Last Maintenance", str(date.today()))

        # Chart for all component probabilities
        st.write("#### Component Probabilities for Selected Machine")
        component_probs = df.loc[df["Machine ID"] == selected_machine, components].melt(
            var_name="Component", value_name="Failure Probability"
        )
        st.bar_chart(component_probs.set_index("Component"))

    else:
        st.info("Please select both Machine ID and Component to view details.")

else:
    st.warning("Please select a date and enter a valid Top N number to proceed.")
