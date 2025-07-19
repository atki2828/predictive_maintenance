import os

import polars as pl

VAL_DATA_PATH = "./data/validation_analysis.csv"


def main():
    val_analysis_df = pl.read_csv(VAL_DATA_PATH)
    print("dummy stop")


if __name__ == "__main__":
    main()
