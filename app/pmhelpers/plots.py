from typing import Dict, List, Optional, Tuple, Union

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
    df: pd.DataFrame,
    x: str,
    y: str,
    figsize: Tuple[int, int] = (15, 5),
    alpha: float = 0.3,
    title: str = "",
    hue: str = "",
) -> plt.Figure:
    """
    Plot a combination of a boxplot and a stripplot for visualizing data distribution and individual observations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to plot.
    x : str
        Column name to use for the x-axis (categorical variable).
    y : str
        Column name to use for the y-axis (numeric variable).
    figsize : tuple of int, default=(15, 5)
        Size of the figure in inches (width, height).
    alpha : float, default=0.3
        Transparency level for the boxplot fill (0.0 transparent, 1.0 opaque).
    title : str, default=""
        Title for the plot.
    hue : str, optional
        Column name for grouping data within x categories (for color separation).
        If not provided, defaults to the value of `x`.

    Returns
    -------
    plt.Figure
        A Matplotlib Figure object containing the combined plot.

    Notes
    -----
    - The boxplot shows median, quartiles, and whiskers without outliers (`showfliers=False`).
    - The stripplot overlays individual data points with jitter for better visibility.

    Example
    -------
    >>> fig = plot_box_and_strip(df, x="component", y="time_to_failure", title="Failure Times by Component")
    >>> fig.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not hue:
        hue = x

    # Boxplot
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

    # Stripplot for individual points
    sns.stripplot(
        data=df,
        x=x,
        y=y,
        color="black",
        size=3.5,
        jitter=True,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()
    return fig


def survival_hazard_group_plotter(
    model_dict: Dict[str, object], model_name: str = ""
) -> plt.Figure:
    """
    Plot survival and hazard functions for multiple fitted survival models grouped by category.

    Parameters
    ----------
    model_dict : Dict[str, object]
        Dictionary where keys represent group names and values are fitted lifelines models
        (e.g., KaplanMeierFitter, WeibullFitter, NelsonAalenFitter).
    model_name : str, optional
        Name of the survival model type (used in plot titles), by default "".

    Returns
    -------
    plt.Figure
        Matplotlib Figure containing two subplots:
        - Top: Survival (Reliability) curves for each group
        - Bottom: Hazard curves for each group

    Notes
    -----
    - The function expects each model in `model_dict` to have `plot_survival_function`
      and `plot_hazard` methods (true for most lifelines univariate models).
    - Colors are automatically assigned by lifelines.

    Example
    -------
    >>> fig = survival_hazard_group_plotter(model_dict, model_name="Kaplan-Meier")
    >>> fig.show()
    """
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 8), sharex=True)

    # Plot survival functions for all groups
    for group, model in model_dict.items():
        model.plot_survival_function(ax=ax[0], label=group)
    ax[0].set_title(f"Reliability Function {model_name}")
    ax[0].set_xlabel("Time (Days)")
    ax[0].set_ylabel("Survival Probability")
    ax[0].legend(title="Group")

    # Plot hazard functions for all groups
    for group, model in model_dict.items():
        model.plot_hazard(ax=ax[1], label=group)
    ax[1].set_title(f"Hazard Function {model_name}")
    ax[1].set_xlabel("Time (Days)")
    ax[1].set_ylabel("Hazard")
    ax[1].legend(title="Group")

    plt.tight_layout()
    return fig


def plot_timeseries_stacked(
    df: Union[pd.DataFrame, pl.DataFrame],
    sensors: List[str],
    time_col: str = "date",
    machine_id: Optional[str] = None,
) -> plt.Figure:
    """
    Plot stacked time-series charts for selected sensors with annotated event markers
    (errors, failures, and maintenance) using Matplotlib and Seaborn.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        Input dataset containing time-series sensor data and event columns.
        Must include the columns specified in `sensors` and `time_col`.
    sensors : List[str]
        List of sensor column names to plot, each in its own subplot.
    time_col : str, default="date"
        Column name representing the timestamp.
    machine_id : str, optional
        If provided, filters the data for a specific machine (requires a `machineID` column).

    Returns
    -------
    plt.Figure
        A Matplotlib Figure object with stacked subplots for each sensor.

    Notes
    -----
    - Event markers included:
        * Errors: Columns ["error1", "error2", "error3", "error4", "error5"]
        * Failures: Columns ["comp1_failure", "comp2_failure", "comp3_failure", "comp4_failure"]
        * Maintenance: Columns ["comp1", "comp2", "comp3", "comp4"]
    - Event line styles:
        * Errors: dotted
        * Failures: solid
        * Maintenance: dashed
    - Converts Polars to Pandas for Seaborn compatibility.

    Example
    -------
    >>> fig = plot_timeseries_stacked(df, sensors=["temperature", "pressure"], machine_id="M123")
    >>> fig.show()
    """
    # Ensure Pandas for plotting
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Optional filter by machine ID
    if machine_id is not None and "machineID" in df.columns:
        df = df[df["machineID"] == machine_id]

    n = len(sensors)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Event styles
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

    # Plot each sensor
    for i, sensor in enumerate(sensors):
        ax = axes[i]
        sns.lineplot(data=df, x=time_col, y=sensor, ax=ax)
        ax.set_title(sensor.replace("_", " ").title())
        ax.set_ylabel("")
        ax.grid(True)

        # Add event lines
        for event_type, style in event_styles.items():
            for col, color in zip(style["columns"], style["colors"]):
                if col in df.columns:
                    times = df.loc[df[col] == 1, time_col]
                    for t in times:
                        ax.axvline(
                            t, color=color, linestyle=style["linestyle"], alpha=0.6
                        )

    # Create legend for events
    legend_lines = []
    for event_type in ["failure", "maintenance", "error"]:  # desired order
        style = event_styles[event_type]
        for col, color in zip(style["columns"], style["colors"]):
            legend_lines.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=style["linestyle"],
                    alpha=0.6,
                    label=col.replace("_", " "),
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


def generate_survival_curv_example_fig() -> plt.Figure:
    """
    Generate an illustrative figure showing:
    1. Lollipop plot of machine lifetimes with failures vs censored observations
    2. Weibull fitted survival curve
    3. Weibull fitted hazard curve

    This function creates synthetic machine failure data using a Weibull distribution,
    fits a Weibull survival model, and visualizes results in three subplots.

    Returns
    -------
    plt.Figure
        A Matplotlib Figure object containing the three plots:
        - Left: Individual machine lifetimes (failures vs censoring)
        - Middle: Survival curve fitted by Weibull model
        - Right: Hazard curve fitted by Weibull model

    Notes
    -----
    - Synthetic dataset uses `n_machines=30` and time horizon of 45 days.
    - Failures are drawn from a Weibull distribution with shape parameter `a=2`.
    - Machines with durations >= max_time are considered censored.

    Example
    -------
    >>> fig = generate_survival_curv_example_fig()
    >>> fig.show()
    """
    # Reproducibility
    np.random.seed(42)

    # Generate synthetic machine failure data
    n_machines = 30
    max_time = 45

    true_failures = np.random.weibull(a=2, size=n_machines) * 35
    event_times = np.clip(true_failures, 0, max_time)
    observed = event_times < max_time

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

    # Fit Weibull survival model
    wf = WeibullFitter()
    wf.fit(df["duration"], event_observed=df["event"])

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1, 1]}
    )

    # ---- Left Plot: Machine lifetimes ----
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

    # ---- Middle Plot: Survival curve ----
    wf.plot_survival_function(ax=ax2)
    ax2.set_title("Weibull Fitted Survival Curve")
    ax2.set_xlabel("Time (Days)")
    ax2.set_ylabel("Survival Probability")

    # ---- Right Plot: Hazard curve ----
    wf.plot_hazard(ax=ax3)
    ax3.set_title("Weibull Fitted Hazard Curve")
    ax3.set_xlabel("Time (Days)")
    ax3.set_ylabel("Hazard")

    fig.tight_layout()
    return fig


def plot_failure_counts(df_plot: pl.DataFrame) -> plt.Figure:
    """
    Plot a bar chart of component failure counts with percentage labels on each bar.

    Parameters
    ----------
    df_plot : pl.DataFrame
        A Polars DataFrame containing at least the following columns:
        - 'failure': Component name or failure type (categorical or string)
        - 'count': Number of failures for that component
        - 'percent': Percentage of total failures for that component

    Returns
    -------
    plt.Figure
        A Matplotlib Figure object containing the bar plot.

    Notes
    -----
    - Converts the Polars DataFrame to Pandas internally for seaborn compatibility.
    - Displays percentage values above each bar for clarity.
    - Rotates x-axis labels by 45 degrees for readability.

    Example
    -------
    >>> fig = plot_failure_counts(failure_counts_df)
    >>> fig.show()
    """
    # Convert Polars to Pandas for seaborn compatibility
    if isinstance(df_plot, pl.DataFrame):
        df_plot = df_plot.to_pandas()

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.barplot(data=df_plot, x="failure", y="count", ax=ax, hue="failure", dodge=False)

    # Add percentage labels above each bar
    for i, row in df_plot.iterrows():
        ax.text(
            i,
            row["count"],
            f"{row['percent']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Customize axes and title
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Failure Component Counts with Percentages", fontsize=14)
    ax.set_xlabel("Component", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    return fig


def plot_time_between_failures_dist(time_to_fail_df: pl.DataFrame) -> plt.Figure:
    """
    Plot the distribution of time between component failures using a histogram with KDE overlay.

    Parameters
    ----------
    time_to_fail_df : pl.DataFrame
        A Polars DataFrame containing the column `time_between_failures`,
        which represents the time difference (in days) between successive failures for components.

    Returns
    -------
    plt.Figure
        A Matplotlib Figure object containing the histogram and KDE.

    Notes
    -----
    - Converts the input Polars DataFrame to pandas for compatibility with seaborn.
    - Adds a kernel density estimate (KDE) overlay for smooth distribution visualization.

    Example
    -------
    >>> fig = plot_time_between_failures_dist(time_to_fail_df)
    >>> fig.show()
    """
    # Convert Polars to Pandas for seaborn compatibility
    pdf = time_to_fail_df.to_pandas()

    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.histplot(pdf, x="time_between_failures", kde=True, ax=ax)

    # Customize plot labels and title
    ax.set_title("Distribution of Time Between Component Failures", fontsize=14)
    ax.set_xlabel("Time Between Failures (Days)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    # Adjust layout for better spacing
    fig.tight_layout()

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
    df: Union[pd.DataFrame, pl.DataFrame],
    sensors: List[str],
    time_col: str = "date",
    machine_id: Optional[str] = None,
) -> go.Figure:
    """
    Create a stacked time-series plot using Plotly for selected sensors, optionally filtered by machine ID.
    Includes additional event annotations for errors, failures, and maintenance activities.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        Input data containing time series and event columns. Can be a Pandas or Polars DataFrame.
    sensors : List[str]
        List of sensor columns to plot on separate subplots.
    time_col : str, default="date"
        Column name representing the timestamp.
    machine_id : str, optional
        If provided, filters the data for a specific machine (requires "machineID" column in df).

    Returns
    -------
    go.Figure
        A Plotly Figure object containing stacked subplots for each sensor and overlaid event markers.

    Notes
    -----
    - Event overlays:
        * Errors: Columns ["error1", "error2", "error3", "error4", "error5"]
        * Failures: Columns ["comp1_failure", "comp2_failure", "comp3_failure", "comp4_failure"]
        * Maintenance: Columns ["comp1", "comp2", "comp3", "comp4"]
    - Uses distinct colors and line styles for each event type:
        * Error: orange, blueviolet, brown, firebrick, violet (dotted)
        * Failure: red, green, black, blue (solid)
        * Maintenance: red, green, black, blue (dashed)

    Example
    -------
    >>> fig = plot_timeseries_stacked_plotly(df, sensors=["temperature", "pressure"], machine_id="M123")
    >>> fig.show()
    """
    # Convert Polars to Pandas if needed
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Filter by machine ID if provided
    if machine_id is not None and "machineID" in df.columns:
        df = df[df["machineID"] == machine_id]

    # Define event styles
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
                showlegend=False,  # legend for events only
            ),
            row=i,
            col=1,
        )

    # Build legend items (one copy per event type/color)
    legend_items = []
    for event_type in ["failure", "maintenance", "error"]:
        style = event_styles[event_type]
        for col, color in zip(style["columns"], style["colors"]):
            legend_items.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=color, dash=style["dash"], width=2),
                    name=col.replace("_", " "),
                )
            )

    # Add all legend traces
    for item in legend_items:
        fig.add_trace(item, row=1, col=1)

    # Add vertical event lines per subplot
    for i, sensor in enumerate(sensors, start=1):
        yref = f"y{i}" if i > 1 else "y"
        for event_type in ["failure", "maintenance", "error"]:
            style = event_styles[event_type]
            for col, color in zip(style["columns"], style["colors"]):
                event_times = df.loc[df[col] == 1, time_col]
                for t in event_times:
                    fig.add_shape(
                        type="line",
                        x0=t,
                        x1=t,
                        y0=df[sensor].min(),
                        y1=df[sensor].max(),
                        xref="x",
                        yref=yref,
                        line=dict(color=color, dash=style["dash"], width=1.5),
                    )

    # Layout for Streamlit look
    fig.update_layout(
        height=260 * n,
        title="Telemetry Time Series",
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Event Legend",
        ),
        margin=dict(l=40, r=200, t=60, b=40),
    )

    # Update axis labels
    fig.update_xaxes(title_text="Date", row=n, col=1)
    fig.update_yaxes(title_text="", showgrid=True)

    return fig
