from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxTimeVaryingFitter, WeibullFitter


def create_val_analysis_df(
    df: pl.DataFrame, model_store: Dict[str, CoxTimeVaryingFitter]
) -> pd.DataFrame:
    val_pd = df.to_pandas().copy()
    for comp, ctv in model_store.items():
        col_name = f"{comp}_failure_proba"
        val_pd[col_name] = predict_failure_proba(val_pd, ctv)
        val_pd[f"{comp}_partial_hazard"] = ctv.predict_partial_hazard(val_pd)
    validation_analysis_df = pl.from_pandas(val_pd)
    return validation_analysis_df


def train_component_models(
    train_df: pl.DataFrame, comp_telemetry_map: Dict[str, str]
) -> Dict[str, CoxTimeVaryingFitter]:
    model_store = dict()

    # Iterate through all components
    for comp in ["comp1", "comp2", "comp3", "comp4"]:
        print(comp)
        formula = f"age + C(model) + {comp_telemetry_map[comp]}"
        print(formula)
        ctv = CoxTimeVaryingFitter(penalizer=0.01)

        ctv.fit(
            df=train_df.to_pandas(),
            event_col=f"{comp}_failure",
            start_col="start",
            stop_col="end",
            id_col="component_instance_id",
            formula=formula,
        )

        model_store[comp] = ctv
    return model_store


def _lookup_baseline_ch(t: int, fitted_cox_model: CoxTimeVaryingFitter):
    if pd.isna(t):
        return 0.0
    baseline_ch = fitted_cox_model.baseline_cumulative_hazard_
    valid_times = baseline_ch.index[baseline_ch.index <= t]
    if valid_times.empty:
        return 0.0
    return baseline_ch.loc[valid_times.max()].iloc[0]


def predict_failure_proba(
    df: pd.DataFrame,
    ctv_model: CoxTimeVaryingFitter,
    end_col: str = "end",
    n_digits: int = 2,
) -> np.ndarray:
    data_baseline_ch = df[end_col].apply(lambda t: _lookup_baseline_ch(t, ctv_model))
    partial_hazard = ctv_model.predict_partial_hazard(df)
    return np.round(1 - np.exp(-data_baseline_ch * partial_hazard), n_digits)


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
