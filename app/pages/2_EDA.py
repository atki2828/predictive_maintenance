import streamlit as st

from pmhelpers.dataprocessing import load_data

telemetry_file_path = "../data/PdM_telemetry.csv"


def main():
    df_tel = load_data(telemetry_file_path)
    st.dataframe(df_tel.head())
