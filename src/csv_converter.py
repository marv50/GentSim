import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import ast
import re
from typing import Tuple, Optional, List


def parse_position(pos_str: str) -> Tuple[int, int]:
    """
    Parse a position string into x, y coordinates.
    Handles strings like "(np.int64(20), np.int64(15))" or plain tuples.
    """
    try:
        # Try extracting numbers wrapped in np.int64(...)
        numbers = re.findall(r'np\.int64\((\d+)\)', pos_str)
        if len(numbers) == 2:
            return int(numbers[0]), int(numbers[1])
        # Fallback: regular tuple
        return tuple(ast.literal_eval(pos_str))
    except Exception:
        raise ValueError(f"Invalid position string: {pos_str}")


def determine_grid_size(df: pd.DataFrame, grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Determine grid size either from user input or based on data.
    """
    if grid_size:
        return grid_size
    return df['y'].max() + 1, df['x'].max() + 1


def create_grid_for_step(step_data: pd.DataFrame, grid_shape: Tuple[int, int], value_column: str) -> np.ndarray:
    """
    Create a single 2D grid for a time step.
    """
    grid = np.zeros(grid_shape)
    for _, row in step_data.iterrows():
        x, y = int(row['x']), int(row['y'])
        if 0 <= y < grid_shape[0] and 0 <= x < grid_shape[1]:
            grid[y, x] = row[value_column]
    return grid


def csv_to_timeseries_grid(csv_file_path: str, value_column: str = 'income',
                           grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Convert CSV agent data to a 3D time series grid.

    Returns: ndarray of shape (time_steps, height, width)
    """
    df = pd.read_csv(csv_file_path)

    # Parse and split position
    df[['x', 'y']] = df['pos'].apply(parse_position).apply(pd.Series)

    # Determine grid shape
    height, width = determine_grid_size(df, grid_size)
    shape = (len(df['Step'].unique()), height, width)

    # Build time series grid
    steps = sorted(df['Step'].unique())
    ts_grid = np.zeros(shape)

    for i, step in enumerate(steps):
        step_data = df[df['Step'] == step]
        ts_grid[i] = create_grid_for_step(step_data, (height, width), value_column)

    return ts_grid


def visualize_grid_evolution(timeseries_grid: np.ndarray, step_indices: Optional[List[int]] = None,
                             figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None):
    """
    Plot the evolution of a 3D time series grid over time.
    """
    if save_path:
        matplotlib.use('Agg')  # Headless rendering

    total_steps = timeseries_grid.shape[0]
    if not step_indices:
        step_indices = list(range(min(6, total_steps)))

    n = len(step_indices)
    cols = min(3, n)
    rows = -(-n // cols)  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for i, step_idx in enumerate(step_indices):
        ax = axes[i // cols, i % cols]
        im = ax.imshow(timeseries_grid[step_idx], cmap='viridis', origin='lower')
        ax.set_title(f"Step {step_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax)

    # Hide any unused axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


# -----------------------
# Example Usage
# -----------------------
if __name__ == "__main__":
    filepath = 'data/agent_data.csv'

    # Grid using income values
    income_grids = csv_to_timeseries_grid(filepath, value_column='income')
    print(f"Income grids shape: {income_grids.shape}")

    # Grid using agent IDs
    agent_grids = csv_to_timeseries_grid(filepath, value_column='AgentID')
    print(f"Agent ID grids shape: {agent_grids.shape}")

    # Stats
    print(f"Grid size: {income_grids.shape[1]} x {income_grids.shape[2]}")
    print(f"Time steps: {income_grids.shape[0]}")
    print(f"Income range: {income_grids.min()} - {income_grids.max()}")

    # Visualize
    visualize_grid_evolution(income_grids, save_path='fig/income_grid_evolution.png')
