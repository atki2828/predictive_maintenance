import os

import streamlit as st
from app_text import (
    feature_text,
    telemetry_component_four_text,
    telemetry_component_one_text,
    telemetry_component_three_text,
    telemetry_component_two_text,
    telemetry_eda_intro,
)

DAILY_PLOT_DATA_PATH = "./app/data/daily_plot_df.csv"
BASE_DIR = "app/"


def main():
    st.set_page_config(layout="wide", page_icon="📈")

    st.markdown(telemetry_eda_intro)
    st.divider()

    comp1_fail_path = os.path.join(BASE_DIR, "static_plots", "comp1_fig.png")
    st.markdown(telemetry_component_one_text)
    st.image(
        comp1_fail_path,
        caption="Comp 1 Telemetry Failure Signal",
        use_container_width=True,
    )
    st.divider()

    comp2_fail_path = os.path.join(BASE_DIR, "static_plots", "comp2_fig.png")
    st.markdown(telemetry_component_two_text)
    st.image(
        comp2_fail_path,
        caption="Comp 1 Telemetry Failure Signal",
        use_container_width=True,
    )
    st.divider()

    comp3_fail_path = os.path.join(BASE_DIR, "static_plots", "comp3_fig.png")
    st.markdown(telemetry_component_three_text)
    st.image(
        comp3_fail_path,
        caption="Comp 3 Telemetry Failure Signal",
        use_container_width=True,
    )
    st.divider()

    comp4_fail_path = os.path.join(BASE_DIR, "static_plots", "comp4_fig.png")
    st.markdown(telemetry_component_four_text)
    st.image(
        comp4_fail_path,
        caption="Comp 4 Telemetry Failure Signal",
        use_container_width=True,
    )
    st.divider()

    st.markdown(feature_text)


if __name__ == "__main__":
    main()
