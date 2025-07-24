import os

import streamlit as st
from app_text import (
    compare_time_between_fail_text,
    component_failure_text,
    eda_intro,
    time_between_comp_fail_text,
    time_between_failure_text,
    weibull_survival_text,
)

BASE_DIR = "app/"


def main():
    # Read and Cache Data
    st.set_page_config(layout="centered", page_icon="🔎")

    # Intro
    st.markdown(eda_intro)

    # Component Failure Plot
    comp_fail_path = os.path.join(BASE_DIR, "static_plots", "failure_count.png")
    st.markdown(component_failure_text)
    st.image(comp_fail_path, caption="Failure Distribution", use_container_width=True)
    st.divider()

    # Time To Failure Distribution
    time_between_comp_fail_dist_path = os.path.join(
        BASE_DIR, "static_plots", "time_between_comp_fail_dist.png"
    )
    st.markdown(time_between_failure_text)
    st.image(
        time_between_comp_fail_dist_path,
        caption="Time Between Failures Distribution",
        use_container_width=True,
    )
    st.divider()

    # Time Between Component Failure (Box + Strip)
    time_between_comp_fail_fig_path = os.path.join(
        BASE_DIR, "static_plots", "time_to_fail_by_comp.png"
    )
    st.markdown(time_between_comp_fail_text)
    st.image(
        time_between_comp_fail_fig_path,
        caption="Time Between Failures by Component",
        use_container_width=True,
    )
    st.divider()

    # Compare Time Between Fail (Box + Strip)
    compare_fig_path = os.path.join(BASE_DIR, "static_plots", "compare_fig.png")
    st.markdown(compare_time_between_fail_text)
    st.image(
        compare_fig_path,
        caption="Time Between Maintenance and Failures",
        use_container_width=True,
    )
    st.divider()

    # Survival and Weibull Analysis
    survival_fig_path = os.path.join(BASE_DIR, "static_plots", "survival.png")
    st.markdown(weibull_survival_text)
    st.image(
        survival_fig_path,
        caption="Weibull Model Survival and Hazard Functions",
        use_container_width=True,
    )
    st.divider()


if __name__ == "__main__":
    main()
