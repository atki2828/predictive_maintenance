import polars as pl
import os


if __name__ == "__main__":
   print(os.getcwd())
   df= pl.read_csv("data/PdM_telemetry.csv")
   print(df.head())