import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib
from typing import List, Optional, Tuple

from src.income_distribution import repeat_data, load_distribution

plt.style.use("bmh")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 18
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
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_grid_evolution(timeseries_grid: np.ndarray,
                             step_indices: Optional[List[int]] = None,
                             figsize: Tuple[int, int] = (15, 10),
                             save_path: Optional[str] = None):
    if save_path:
        matplotlib.use('Agg')

    total_steps = timeseries_grid.shape[0]
    if step_indices is None:
        num_snapshots = min(6, total_steps)
        step_indices = np.linspace(0, total_steps - 1, num=num_snapshots, dtype=int).tolist()


    n = len(step_indices)
    cols = min(3, n)
    rows = -(-n // cols)  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    import matplotlib.colors as mcolors

    # Define the bins
    bounds = [0, 1, 38690, 77280, 100000]
    labels = ['empty', 'low', 'medium', 'high']

    # Create a ListedColormap and BoundaryNorm
    cmap = mcolors.ListedColormap(['#000000','#1f77b4', '#2ca02c', '#ff7f0e'])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for i, step_idx in enumerate(step_indices):
        ax = axes[i // cols, i % cols]
        im = ax.imshow(timeseries_grid[step_idx],
                       cmap=cmap, norm=norm, origin='lower')
        ax.set_title(f"Step {step_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
