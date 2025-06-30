from typing import List, Optional, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.income_distribution import load_distribution, repeat_data

plt.style.use("bmh")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 20
FIG_DPI = 300


def plot_income_distribution(
    title: str = "Income Distribution",
    xlabel: str = "Income",
    ylabel: str = "Frequency",
    bins: int = 50,
    save=True,
    save_path: str = "fig/income_distribution.png",
):
    """
    Plot the income distribution.
    """
    df = load_distribution()

    expanded_data = repeat_data(df)
    pd.Series(expanded_data).plot(
        kind="hist", bins=bins, edgecolor="black", density=True
    )
    plt.title(title, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_grid_evolution(
    timeseries_grid: np.ndarray,
    n_houses: int,
    income_bounds: Optional[List[int]] = None,
    step_indices: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
):
    if save_path:
        matplotlib.use("Agg")

    total_steps = timeseries_grid.shape[0]
    if step_indices is None:
        num_snapshots = min(6, total_steps)
        step_indices = np.linspace(
            0, total_steps - 1, num=num_snapshots, dtype=int
        ).tolist()

    n = len(step_indices)
    cols = min(3, n)
    rows = -(-n // cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    cmap = mcolors.ListedColormap(["#000000", "#1f77b4", "#2ca02c", "#ff7f0e"])
    income_bounds.insert(0, 1)
    norm = mcolors.BoundaryNorm(income_bounds, cmap.N)

    for i, step_idx in enumerate(step_indices):
        ax = axes[i // cols, i % cols]
        grid = timeseries_grid[step_idx]
        height, width = grid.shape

        _ = ax.imshow(
            grid,
            cmap=cmap,
            norm=norm,
            origin="lower",
            extent=[0, width, 0, height],
            interpolation="none",
        )

        ax.set_xticks(np.arange(0, width + 1, n_houses))
        ax.set_yticks(np.arange(0, height + 1, n_houses))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(which="both", color="white", linestyle="--", linewidth=1)

        ax.set_title(f"Step {step_idx + 1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    legend_elements = [
        Patch(facecolor="#000000", label="empty houses"),
        Patch(facecolor="#1f77b4", label="low income"),
        Patch(facecolor="#2ca02c", label="medium income"),
        Patch(facecolor="#ff7f0e", label="high income"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.18, 0.9),
        frameon=True,
        fontsize="medium",
    )
    fig.suptitle("Evolution of Houseold Movement", fontsize=25, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_spatial_disparity_over_time(
    disparity_values, uncertainty=None, output_path="fig/disparity_over_time.png"
):
    """
    Plots and saves the average spatial income disparity over time, with optional uncertainty shading.

    Parameters:
    - disparity_values (list or np.array): The average disparity values over time (1D).
    - uncertainty (list or np.array, optional): The std or SEM for each time step (same length as disparity_values).
    - output_path (str): File path to save the plot (including filename).
    """
    disparity_values = np.array(disparity_values)

    plt.figure(figsize=(8, 5))
    plt.plot(disparity_values, label="Mean Disparity", color="blue")

    if uncertainty is not None:
        uncertainty = np.array(uncertainty)
        lower = disparity_values - uncertainty
        upper = disparity_values + uncertainty
        plt.fill_between(
            np.arange(len(disparity_values)),
            lower,
            upper,
            color="blue",
            alpha=0.3,
            label="Uncertainty",
        )

    plt.xlabel("Time Step")
    plt.ylabel("Income Disparity")
    plt.title("Average Spatial Income Disparity Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_elementary_effects(data, parameters):
    """
    Plots mu_star and sigma from Morris elementary effects results.

    Parameters:
    - data (dict or pd.DataFrame): Should contain 'mu_star' and 'sigma' values.
    - parameters (list): List of parameter names corresponding to data.
    """
    # If input data is dict, convert to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame(data, index=parameters)
    else:
        df = data.copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    bar_width = 0.35

    # Plot bars
    ax.bar(
        x - bar_width / 2,
        df["mu_star"],
        width=bar_width,
        label="μ★ (mu_star)",
        color="darkorange",
    )
    ax.bar(
        x + bar_width / 2,
        df["sigma"],
        width=bar_width,
        label="σ (sigma)",
        color="slateblue",
    )

    # Labels and styling
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Effect Value")
    ax.set_title("Morris Sensitivity Analysis: μ★ and σ")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig/morris_sensitivity_analysis.png", dpi=FIG_DPI)
    plt.close()
