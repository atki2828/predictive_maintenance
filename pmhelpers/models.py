from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxTimeVaryingFitter, WeibullFitter


def create_val_analysis_df(
    df: pl.DataFrame, model_store: Dict[str, CoxTimeVaryingFitter]
) -> pl.DataFrame:
    """
    Generate a validation analysis DataFrame by applying Cox Time-Varying models to compute
    failure probabilities and partial hazards for each component.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame containing the validation dataset.
        This should include all covariates required by the Cox models in `model_store`.

    model_store : Dict[str, CoxTimeVaryingFitter]
        Dictionary mapping component names to trained CoxTimeVaryingFitter models.
        Example: {"comp1": CoxTimeVaryingFitter(...), "comp2": ...}

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with additional columns for:
        - `<component>_failure_proba`: Predicted failure probability for the component.
        - `<component>_partial_hazard`: Partial hazard value predicted by the model.

    Notes
    -----
    - The function converts the input Polars DataFrame to pandas internally to work
      with `lifelines` models, then converts back to Polars.
    - `predict_failure_proba` should be a custom function that computes survival-based
      failure probability using the Cox model output.

    Example
    -------
    >>> validation_df = create_val_analysis_df(df, model_store)
    >>> validation_df.head()
    shape: (n, m)  # original columns + failure_proba and partial_hazard per component
    """
    # Convert Polars to pandas for lifelines
    val_pd = df.to_pandas().copy()

    # Compute predictions for each component
    for comp, ctv in model_store.items():
        col_name = f"{comp}_failure_proba"
        val_pd[col_name] = predict_failure_proba(val_pd, ctv)
        val_pd[f"{comp}_partial_hazard"] = ctv.predict_partial_hazard(val_pd)

    # Convert back to Polars
    validation_analysis_df = pl.from_pandas(val_pd)
    return validation_analysis_df


def train_component_models(
    train_df: pl.DataFrame, comp_telemetry_map: Dict[str, str]
) -> Dict[str, CoxTimeVaryingFitter]:
    """
    Train Cox Time-Varying models for multiple components using a provided training dataset.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training dataset in Polars format. Must include:
        - `start` and `end` columns for time intervals.
        - `component_instance_id` to identify individual components.
        - Failure indicator columns for each component (e.g., `comp1_failure`, `comp2_failure`).
        - Features such as `age`, `model`, and telemetry variables from `comp_telemetry_map`.

    comp_telemetry_map : Dict[str, str]
        A dictionary mapping component names to their associated telemetry feature.
        Example: {"comp1": "pressure", "comp2": "temperature"}

    Returns
    -------
    Dict[str, CoxTimeVaryingFitter]
        A dictionary of trained CoxTimeVaryingFitter models keyed by component name.
        Example: {"comp1": CoxTimeVaryingFitter(...), "comp2": ...}

    Notes
    -----
    - Uses a penalizer of 0.01 for regularization.
    - Converts `train_df` to pandas internally for compatibility with lifelines.
    - The formula includes `age`, categorical encoding of `model`, and the component-specific telemetry feature.

    Example
    -------
    >>> comp_telemetry_map = {"comp1": "pressure", "comp2": "temperature", "comp3": "vibration", "comp4": "voltage"}
    >>> models = train_component_models(train_df, comp_telemetry_map)
    >>> models["comp1"].print_summary()
    """
    model_store: Dict[str, CoxTimeVaryingFitter] = {}

    for comp in ["comp1", "comp2", "comp3", "comp4"]:
        print(f"Training model for: {comp}")
        formula = f"age + C(model) + {comp_telemetry_map[comp]}"
        print(f"Formula: {formula}")

        # Initialize Cox Time-Varying model with L2 regularization
        ctv = CoxTimeVaryingFitter(penalizer=0.01)

        # Fit the model using pandas DataFrame
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


def _lookup_baseline_ch(t: int, fitted_cox_model: CoxTimeVaryingFitter) -> float:
    """
    Look up the baseline cumulative hazard at or before a given time `t`
    from a fitted Cox Time-Varying model.

    Parameters
    ----------
    t : int
        The time value at which to look up the baseline cumulative hazard.
        If `t` is NaN, the function returns 0.0.

    fitted_cox_model : CoxTimeVaryingFitter
        A trained Cox Time-Varying model from lifelines, which contains
        `baseline_cumulative_hazard_` as a Pandas DataFrame.

    Returns
    -------
    float
        The cumulative hazard value at the closest time less than or equal to `t`.
        Returns 0.0 if `t` is NaN or if no valid times are found in the baseline hazard.

    Notes
    -----
    - The baseline cumulative hazard DataFrame typically has:
        index: time values
        columns: hazard values (usually one column for all strata)
    - If `t` falls between two times, the function selects the largest time
      less than or equal to `t`.

    Example
    -------
    >>> value = _lookup_baseline_ch(50, cox_model)
    >>> print(value)  # e.g., 0.12345
    """
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
    """
    Predict failure probabilities for each observation using a fitted Cox Time-Varying model.

    The failure probability is computed as:
        P(event by time t) = 1 - exp(-H0(t) * exp(Xβ))
    where:
        - H0(t) is the baseline cumulative hazard at time t
        - exp(Xβ) is the partial hazard predicted by the Cox model

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing covariates required by the Cox model.
        Must include the time column specified by `end_col`.

    ctv_model : CoxTimeVaryingFitter
        A trained Cox Time-Varying model from lifelines.

    end_col : str, default="end"
        Column name in `df` representing the stop time for the observation.

    n_digits : int, default=2
        Number of decimal places to round the predicted probabilities.

    Returns
    -------
    np.ndarray
        An array of predicted failure probabilities rounded to `n_digits`.

    Notes
    -----
    - Uses the baseline cumulative hazard from the fitted Cox model.
    - Assumes `baseline_cumulative_hazard_` in the model has a single column.
    - Calls `_lookup_baseline_ch` to retrieve H0(t) for each time.

    Example
    -------
    >>> probs = predict_failure_proba(validation_df, ctv_model)
    >>> print(probs[:5])  # e.g., [0.12, 0.34, 0.05, ...]
    """
    data_baseline_ch = df[end_col].apply(lambda t: _lookup_baseline_ch(t, ctv_model))
    partial_hazard = ctv_model.predict_partial_hazard(df)
    return np.round(1 - np.exp(-data_baseline_ch * partial_hazard), n_digits)


def univariate_survival_model_fitter(
    df: pd.DataFrame, model: Any, duration_col: str, observed_col: str, label: str = ""
) -> Any:
    """
    Fit a univariate survival model to the provided dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing survival data.
    model : Any
        An instance of a lifelines univariate survival model (e.g., KaplanMeierFitter or NelsonAalenFitter).
    duration_col : str
        Name of the column representing durations or time-to-event.
    observed_col : str
        Name of the column representing the event indicator (1 if event occurred, 0 if censored).
    label : str, optional
        Label for the fitted model, used for plotting and identification.

    Returns
    -------
    Any
        The fitted survival model (e.g., KaplanMeierFitter or NelsonAalenFitter).

    Example
    -------
    >>> kmf = univariate_survival_model_fitter(df, KaplanMeierFitter(), "time", "event", label="Component A")
    >>> kmf.survival_function_.head()
    """
    durations = df[duration_col]
    event_observed = df[observed_col]
    return model.fit(durations=durations, event_observed=event_observed, label=label)


def create_survival_model_dict(
    fail_flag_df: pl.DataFrame,
    model: Callable[[], Any],
    group_level_col: str,
    duration_col: str,
    observe_col: str,
) -> Dict[str, Any]:
    """
    Fit survival models for each group in a Polars DataFrame and store them in a dictionary.

    Parameters
    ----------
    fail_flag_df : pl.DataFrame
        Polars DataFrame containing survival data for multiple groups.
    model : Callable[[], Any]
        A callable that returns a new instance of a lifelines survival model (e.g., KaplanMeierFitter).
    group_level_col : str
        Column name to group by (e.g., component type).
    duration_col : str
        Name of the column representing durations or time-to-event.
    observe_col : str
        Name of the column representing the event indicator (1 if event occurred, 0 if censored).

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping each group to its fitted survival model.

    Example
    -------
    >>> model_dict = create_survival_model_dict(
    ...     fail_flag_df,
    ...     model=KaplanMeierFitter,
    ...     group_level_col="component",
    ...     duration_col="time",
    ...     observe_col="event"
    ... )
    >>> model_dict["comp1"].plot_survival_function()
    """
    model_dict: Dict[str, Any] = {}
    pandas_df = fail_flag_df.to_pandas()

    for group in pandas_df[group_level_col].unique():
        group_df = pandas_df[pandas_df[group_level_col] == group]
        model_dict[group] = univariate_survival_model_fitter(
            df=group_df,
            model=model(),
            duration_col=duration_col,
            observed_col=observe_col,
            label=group,
        )

    return model_dict
