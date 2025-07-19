import os
from datetime import datetime

import matplotlib.pyplot as plt
import polars as pl
from lifelines import CoxTimeVaryingFitter

from pmhelpers.dataprocessing import create_train_test_df
from pmhelpers.models import create_val_analysis_df, train_component_models
from pmhelpers.plots import plot_timeseries_stacked

plt.style.use("ggplot")

SPLIT_DATE = datetime(2015, 6, 30)
MODEL_DATA_PATH = "./data/model_data.csv"
WRITE_DIR = "./data/"
TELEMETRY_COLS = [
    "anomaly_flag_21d_mean_daily_voltage",
    "anomaly_flag_21d_mean_daily_rotation",
    "anomaly_flag_21d_mean_daily_pressure",
    "anomaly_flag_21d_mean_daily_vibration",
]


COMP_TEL_COL_MAP = {
    "comp1": "anomaly_flag_21d_mean_daily_voltage",
    "comp2": "anomaly_flag_21d_mean_daily_rotation",
    "comp3": "anomaly_flag_21d_mean_daily_pressure",
    "comp4": "anomaly_flag_21d_mean_daily_vibration",
}


def main():
    print("Running")
    model_df = pl.read_csv(MODEL_DATA_PATH, try_parse_dates=True)
    train_df, val_df = create_train_test_df(
        df=model_df, split_date=SPLIT_DATE, shift_telem_days=2
    )
    print("Training Models")
    model_store = train_component_models(
        train_df=train_df, comp_telemetry_map=COMP_TEL_COL_MAP
    )
    validation_analysis_df = create_val_analysis_df(df=val_df, model_store=model_store)
    validation_analysis_df.write_csv(os.path.join(WRITE_DIR, "validation_analysis.csv"))
    print("fin")


if __name__ == "__main__":
    main()
