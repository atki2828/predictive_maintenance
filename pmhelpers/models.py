import pandas as pd
import polars as pl
from lifelines import WeibullFitter


def univariate_survival_model_fitter(df, model, duration_col, observed_col, label=""):
    durations = df[duration_col]
    event_observed = df[observed_col]
    return (
        model.fit(durations=durations, event_observed=event_observed, label=label)
        if label
        else model.fit(durations=durations, event_observed=event_observed, label=label)
    )


def create_survival_model_dict(
    fail_flag_df, model, group_level_col, duration_col, observe_col
):
    model_dict = dict()
    for group in fail_flag_df.to_pandas()[group_level_col].unique():
        group_df = fail_flag_df.filter(pl.col(group_level_col) == group).to_pandas()
        model_dict[group] = univariate_survival_model_fitter(
            df=group_df,
            model=model(),
            duration_col=duration_col,
            observed_col=observe_col,
            label=group,
        )
    return model_dict
