import matplotlib.pyplot as plt
import polars as pl
import streamlit as st
from app_text import (
    all_machine_failures_text,
    compare_time_between_fail_text,
    component_failure_text,
    eda_intro,
    time_between_comp_fail_text,
    time_between_failure_text,
    weibull_survival_text,
)
from lifelines import WeibullFitter
from pmhelpers.dataprocessing import load_data
from pmhelpers.models import create_survival_model_dict
from pmhelpers.plots import (
    plot_box_and_strip,
    plot_failure_counts,
    plot_time_between_failures_dist,
    survival_hazard_group_plotter,
)

plt.style.use("ggplot")
TELEMETRY_FILE_PATH = "./app/data/PdM_telemetry.csv"
FAILURE_FILE_PATH = "./app/data/PdM_failures.csv"
MACHINE_FILE_PATH = "./app/data/PdM_machines.csv"
MAINTENANCE_FILE_PATH = "./app/data/PdM_maintenance.csv"
ERROR_FILE_PATH = "./app/data/PdM_errors.csv"

COMP_FAIL_PLOT_PATH = "./app/data/comp_fail_plot.csv"
MACHINE_FAIL_PLOT_PATH = "./app/data/machine_fail_plot.csv"
TIME_BETWEEN_FAIL_FLAG_PATH = "./app/data/time_between_fail_flag.csv"
TIME_TO_FAIL_PATH = "./app/data/time_to_fail.csv"


def main():
    # Read and Cache Data
    st.set_page_config(layout="centered", page_icon="🔎")
    comp_fail_plot_df = load_data(COMP_FAIL_PLOT_PATH)
    mach_fail_plot_df = load_data(MACHINE_FAIL_PLOT_PATH)
    time_to_fail_df = load_data(TIME_TO_FAIL_PATH)
    time_between_comp_fail_flag_df = load_data(TIME_BETWEEN_FAIL_FLAG_PATH)
    component_weibull_dict = create_survival_model_dict(
        fail_flag_df=time_between_comp_fail_flag_df,
        model=WeibullFitter,
        group_level_col="model",
        duration_col="time_between_maintenance",
        observe_col="fail_flag",
    )

    # Intro
    st.markdown(eda_intro)

    # Component Failure Plot
    failure_count_fig = plot_failure_counts(comp_fail_plot_df)
    st.markdown(component_failure_text)
    st.pyplot(failure_count_fig)
    st.divider()

    # Time To Failure dist
    st.markdown(time_between_failure_text)
    time_between_fail_dist_fig = plot_time_between_failures_dist(time_to_fail_df)
    st.pyplot(time_between_fail_dist_fig)
    st.divider()

    # Time Between Component Failure Text
    st.markdown(time_between_comp_fail_text)
    time_between_comp_fail_fig = plot_box_and_strip(
        df=time_to_fail_df.sort("failure").to_pandas(),
        x="failure",
        y="time_between_failures",
        title="Time Between Failures By Component",
    )
    st.pyplot(time_between_comp_fail_fig)
    st.divider()

    # Compare Time Between Fail Text
    st.markdown(compare_time_between_fail_text)
    compare_fig = plot_box_and_strip(
        time_between_comp_fail_flag_df.to_pandas(),
        x="fail_flag",
        y="time_between_maintenance",
        title="Time Between Maintenance and Failures",
    )
    st.pyplot(compare_fig)
    st.divider()

    # Survival and Weibull Analysis DF
    st.markdown(weibull_survival_text)
    survival_fig = survival_hazard_group_plotter(
        component_weibull_dict, model_name="Weibull Model"
    )
    st.pyplot(survival_fig)


if __name__ == "__main__":
    main()
