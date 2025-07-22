import matplotlib.pyplot as plt
import streamlit as st
from app_text import theory_cox_models, theory_intro

plt.style.use("ggplot")
from pmhelpers.plots import generate_survival_curv_example_fig


def main():
    st.set_page_config(layout="wide", page_icon="📕")
    st.markdown(theory_intro)
    st.divider()
    survival_example_fig = generate_survival_curv_example_fig()
    st.pyplot(survival_example_fig)
    st.divider()
    st.markdown(theory_cox_models)


if __name__ == "__main__":
    main()
