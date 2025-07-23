import os

import polars as pl

from pmhelpers.dataprocessing import (
    create_comp_dash_demo_df,
    create_fail_prob_dash_demo_df,
    create_telemetry_dash_demo_df,
)

VAL_DATA_PATH = "./data/validation_analysis.csv"
WRITE_DIR = "./data"


def main():
    print("Running Main")
    validation_analysis_df = pl.read_csv(VAL_DATA_PATH)
    telem_df = create_telemetry_dash_demo_df(df=validation_analysis_df)
    comp_df = create_comp_dash_demo_df(df=validation_analysis_df)
    fail_prob_df = create_fail_prob_dash_demo_df(df=validation_analysis_df)

    dash_demo_df = (
        telem_df.join(comp_df, on=["machineID", "date"], how="inner")
        .join(fail_prob_df, on=["machineID", "date"], how="inner")
        .sort(["machineID", "date"])
    )
    print(f"Writing Dash Demo Df \n With shape {dash_demo_df.shape}")
    dash_demo_df.write_csv(os.path.join(WRITE_DIR, "dash_demo.csv"))


if __name__ == "__main__":
    main()
