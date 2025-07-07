from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from matplotlib.lines import Line2D


def box_and_strip(df, x, y, figsize=(15, 5), alpha=0.3, title=""):
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df, x=x, y=y, showcaps=True, boxprops=dict(alpha=alpha), showfliers=False
    )
    sns.stripplot(data=df, x=x, y=y, color="black", size=3.5, jitter=True)

    plt.title(title)
    plt.tight_layout()
    plt.show()


def survival_hazard_group_plotter(model_dict: dict, model_name: str = ""):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))
    for group in model_dict.keys():
        model_dict[group].plot_survival_function(ax=ax[0])
        ax[0].set_title(f"Reliability Function {model_name}")
        ax[0].set_xlabel("Time (Days)")
        ax[0].set_ylabel("Survival Probability")

        model_dict[group].plot_hazard(ax=ax[1])
        ax[1].set_title(f"Hazard Function {model_name}")
        ax[1].set_xlabel("Time (Days)")
        ax[1].set_ylabel("Hazard")
    plt.tight_layout()
    plt.show()


def plot_timeseries_stacked(
    df: pd.DataFrame, sensors: list, time_col="date", machine_id=None
):
    n = len(sensors)

    if machine_id is not None and "machineID" in df.columns:
        df = df[df["machineID"] == machine_id]

    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Define all events and styles in one place
    event_styles = {
        "error": {
            "columns": ["error1", "error2", "error3", "error4", "error5"],
            "colors": ["orange", "blueviolet", "brown", "firebrick", "violet"],
            "linestyle": "dotted",
        },
        "failure": {
            "columns": [
                "comp1_failure",
                "comp2_failure",
                "comp3_failure",
                "comp4_failure",
            ],
            "colors": ["red", "green", "black", "blue"],
            "linestyle": "solid",
        },
        "maintenance": {
            "columns": ["comp1", "comp2", "comp3", "comp4"],
            "colors": ["red", "green", "black", "blue"],
            "linestyle": "dashed",
        },
    }

    for i, sensor in enumerate(sensors):
        ax = axes[i]
        sns.lineplot(data=df, x=time_col, y=sensor, ax=ax)
        ax.set_title(sensor.replace("_", " ").title())
        ax.set_ylabel("")
        ax.grid(True)

        for event_type, style in event_styles.items():
            for column, color in zip(style["columns"], style["colors"]):
                times = df.loc[df[column] == 1, time_col]
                for t in times:
                    ax.axvline(t, color=color, linestyle=style["linestyle"], alpha=0.6)

    legend_lines = []
    for event_type in ["failure", "maintenance", "error"]:  # ensure desired order
        style = event_styles[event_type]
        for column, color in zip(style["columns"], style["colors"]):
            label = column.replace("_", " ")
            legend_lines.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=style["linestyle"],
                    alpha=0.6,
                    label=label,
                )
            )

    axes[0].legend(
        handles=legend_lines, loc="upper left", ncol=1, bbox_to_anchor=(1.01, 0)
    )
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()


def plot_top_cox_predictors(
    model: Union[CoxPHFitter, CoxTimeVaryingFitter],
    top_n: int = 20,
    title: str = "Top Predictors of Hazard (Cox Coefficients)",
):
    """
    Plots the top N most significant predictors from a fitted Cox model.

    Parameters:
    - model: Fitted lifelines CoxTimeVaryingFitter or CoxPHFitter
    - top_n: Number of top predictors to display (ranked by lowest p-value)
    - title: Plot title
    """
    summary = model.summary.sort_values("p")
    top = summary.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top.index[::-1], top["coef"][::-1], xerr=top["se(coef)"][::-1])
    plt.axvline(0, color="black", linestyle="--")
    plt.title(title)
    plt.xlabel("Coefficient (log hazard ratio)")
    plt.tight_layout()
    plt.show()
