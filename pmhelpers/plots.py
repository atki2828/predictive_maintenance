from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, WeibullFitter
from matplotlib.lines import Line2D
from plotly.subplots import make_subplots


def plot_box_and_strip(
    df: pd.DataFrame, x, y, figsize=(15, 5), alpha=0.3, title="", hue=""
):
    fig, ax = plt.subplots(figsize=figsize)
    if not hue:
        hue = x
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        showcaps=True,
        boxprops=dict(alpha=alpha),
        showfliers=False,
        ax=ax,
        hue=hue,
    )
    sns.stripplot(data=df, x=x, y=y, color="black", size=3.5, jitter=True, ax=ax)

    ax.set_title(title)
    plt.tight_layout()
    return fig


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
    return fig


def plot_timeseries_stacked(
    df: pd.DataFrame, sensors: list, time_col="date", machine_id=None
):
    n = len(sensors)
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

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
        handles=legend_lines,
        loc="upper left",
        ncol=1,
        bbox_to_anchor=(1.01, 0),
        title="Telemetry Event Legend",
    )
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return fig


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

    plt.figure(figsize=(12, 6))
    plt.barh(top.index[::-1], top["coef"][::-1], xerr=top["se(coef)"][::-1])
    plt.axvline(0, color="black", linestyle="--")
    plt.title(title)
    plt.xlabel("Coefficient (log hazard ratio)")
    plt.tight_layout()
    plt.show()


def generate_survival_curv_example_fig():
    np.random.seed(42)

    n_machines = 30
    max_time = 45

    true_failures = np.random.weibull(a=2, size=n_machines) * 35
    event_times = np.clip(true_failures, 0, max_time)
    observed = np.where(event_times < max_time, True, False)

    df = (
        pd.DataFrame(
            {
                "machine_id": [f"Machine {i+1}" for i in range(n_machines)],
                "duration": event_times,
                "event": observed,
            }
        )
        .sort_values(by="duration", ascending=True)
        .reset_index(drop=True)
    )

    wf = WeibullFitter()
    wf.fit(df["duration"], event_observed=df["event"])

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1, 1]}
    )

    # Left plot: lollipop failure/censored plot
    for i, row in df.iterrows():
        ax1.hlines(y=i, xmin=0, xmax=row["duration"], color="gray", alpha=0.6)
        marker = "x" if row["event"] else "o"
        color = "red" if row["event"] else "blue"
        ax1.plot(row["duration"], i, marker=marker, color=color, markersize=8)

    ax1.set_yticks(range(n_machines))
    ax1.set_yticklabels(df["machine_id"])
    ax1.set_xlabel("Time to Failure or Censoring (Days)")
    ax1.set_title("Machine Lifetimes: Failures (X) vs Censored (O)")
    ax1.invert_yaxis()

    # Right plot: fitted survival curve
    wf.plot_survival_function(ax=ax2)
    ax2.set_title("Weibull Fitted Survival Curve")
    ax2.set_xlabel("Time (Days)")
    ax2.set_ylabel("Survival Probability")

    wf.plot_hazard(ax=ax3)
    ax3.set_title("Weibull Fitted Hazard Curve")
    ax3.set_xlabel("Time (Days)")
    ax3.set_ylabel("Hazard")
    return fig


def plot_failure_counts(df_plot: pl.DataFrame):
    if isinstance(df_plot, pl.DataFrame):
        df_plot = df_plot.to_pandas()
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.barplot(data=df_plot, x="failure", y="count", ax=ax, hue="failure")
    for i, row in df_plot.iterrows():
        ax.text(i, row["count"], f"{row['percent']:.1f}%", ha="center", va="bottom")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Failure Component Counts with Percentages")
    ax.set_xlabel("Component")
    ax.set_ylabel("Count")
    plt.tight_layout()

    return fig


def plot_machine_failure_counts(df_plot: pl.DataFrame):
    if isinstance(df_plot, pl.DataFrame):
        df_plot = df_plot.to_pandas().assign(
            machineID=lambda df: df["machineID"].astype("category")
        )
    # Create plot
    fig, ax = plt.subplots(figsize=(18, 5))
    sns.barplot(data=df_plot, x="machineID", y="count", ax=ax, hue="machineID")
    # Style the plot
    ax.set_title("Machine ID Failure Counts")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel("Machine")
    ax.set_ylabel("Count")
    ax.get_legend().remove()
    plt.tight_layout()

    return fig


def plot_time_between_failures_dist(time_to_fail_df: pl.DataFrame) -> plt.Figure:

    pdf = time_to_fail_df.to_pandas()

    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.histplot(pdf, x="time_between_failures", kde=True, ax=ax)
    ax.set_title("Distribution of Time Between Component Failures", fontsize=14)
    ax.set_xlabel("Time Between Failures (Days)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    fig.tight_layout()

    return fig


def plot_time_between_maintenance_dist(df: pl.DataFrame) -> plt.Figure:
    """
    Plot and return a histogram of time between maintenance and failure, with hue based on fail_flag.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame with columns:
        - 'time_between_maintenance' (numeric)
        - 'fail_flag' (categorical or boolean)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object for rendering or saving.
    """
    # Convert to pandas for seaborn
    pdf = df.to_pandas()

    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.histplot(
        pdf, x="time_between_maintenance", kde=True, bins=30, hue="fail_flag", ax=ax
    )

    ax.set_title("Time Between Maintenance And Failure", fontsize=14)
    ax.set_xlabel("Time Between Maintenance (Days)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    fig.tight_layout()
    return fig


def plot_timeseries_stacked_plotly(
    df: pd.DataFrame, sensors: list, time_col="date", machine_id=None
):
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    if machine_id is not None and "machineID" in df.columns:
        df = df[df["machineID"] == machine_id]

    # Event style definitions
    event_styles = {
        "error": {
            "columns": ["error1", "error2", "error3", "error4", "error5"],
            "colors": ["orange", "blueviolet", "brown", "firebrick", "violet"],
            "dash": "dot",
        },
        "failure": {
            "columns": [
                "comp1_failure",
                "comp2_failure",
                "comp3_failure",
                "comp4_failure",
            ],
            "colors": ["red", "green", "black", "blue"],
            "dash": "solid",
        },
        "maintenance": {
            "columns": ["comp1", "comp2", "comp3", "comp4"],
            "colors": ["red", "green", "black", "blue"],
            "dash": "dash",
        },
    }

    # Create stacked subplots
    n = len(sensors)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[s.replace("_", " ").title() for s in sensors],
    )

    # Add sensor time series
    for i, sensor in enumerate(sensors, start=1):
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[sensor],
                mode="lines",
                name=sensor.replace("_", " ").title(),
                line=dict(color="#1f77b4", width=2),
            ),
            row=i,
            col=1,
        )

    # Add vertical event lines
    legend_items = []
    for event_type in ["failure", "maintenance", "error"]:  # order matters
        style = event_styles[event_type]
        for col, color in zip(style["columns"], style["colors"]):
            event_times = df.loc[df[col] == 1, time_col]
            for t in event_times:
                fig.add_vline(
                    x=t.strftime("%Y-%m-%d"),
                    line_width=1.5,
                    line_dash=style["dash"],
                    line_color=color,
                    annotation_text=None,
                )
            # Add legend item (fake trace for legend)
            legend_items.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=color, dash=style["dash"], width=2),
                    name=col.replace("_", " "),
                )
            )

    # Add legend items as invisible traces
    for item in legend_items:
        fig.add_trace(item, row=1, col=1)

    # Layout settings for Streamlit look
    fig.update_layout(
        height=300 * n,
        title="Telemetry Time Series",
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Telemetry Event Legend",
        ),
        margin=dict(l=40, r=200, t=60, b=40),
    )

    # Update axis labels
    fig.update_xaxes(title_text="Date", row=n, col=1)
    fig.update_yaxes(title_text="", showgrid=True)

    return fig
