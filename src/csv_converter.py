import numpy as np
import pandas as pd
import ast
import re
from typing import Tuple, Optional, List


def parse_position(pos_str: str) -> Tuple[int, int]:
    try:
        numbers = re.findall(r'np\.int64\((\d+)\)', pos_str)
        if len(numbers) == 2:
            return int(numbers[0]), int(numbers[1])
        numbers = re.findall(r'\d+', pos_str)
        if len(numbers) == 2:
            return int(numbers[0]), int(numbers[1])
        cleaned = re.sub(r'np\.int64\((\d+)\)', r'\1', pos_str)
        return tuple(ast.literal_eval(cleaned))
    except Exception as e:
        raise ValueError(f"Invalid position string: {pos_str}. Error: {e}")


def determine_grid_size(df: pd.DataFrame, grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    return grid_size if grid_size else (df['y'].max() + 1, df['x'].max() + 1)


def create_grid_for_step(step_data: pd.DataFrame, grid_shape: Tuple[int, int], value_column: str) -> np.ndarray:
    grid = np.zeros(grid_shape)
    for _, row in step_data.iterrows():
        x, y = int(row['x']), int(row['y'])
        if 0 <= y < grid_shape[0] and 0 <= x < grid_shape[1]:
            grid[y, x] = row[value_column]
    return grid


def df_to_timeseries_grid(df: pd.DataFrame, value_column: str = 'income',
                          grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    df = df.copy()
    try:
        df[['x', 'y']] = df['pos'].apply(parse_position).apply(pd.Series)
    except Exception:
        def safe_parse(pos_str):
            try:
                return parse_position(pos_str)
            except:
                return (0, 0)
        df[['x', 'y']] = df['pos'].apply(safe_parse).apply(pd.Series)

    height, width = determine_grid_size(df, grid_size)
    steps = sorted(df['Step'].unique())
    ts_grid = np.zeros((len(steps), height, width))

    for i, step in enumerate(steps):
        step_data = df[df['Step'] == step]
        ts_grid[i] = create_grid_for_step(step_data, (height, width), value_column)

    return ts_grid


def csv_to_timeseries_grid(csv_file_path: str, value_column: str = 'income',
                           grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    df = pd.read_csv(csv_file_path)
    return df_to_timeseries_grid(df, value_column, grid_size)


def detect_runs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    step_diff = df['Step'].diff()
    reset_indices = step_diff[step_diff < 0].index.tolist()
    run_boundaries = [0] + reset_indices + [len(df)]
    runs = [(run_boundaries[i], run_boundaries[i + 1]) for i in range(len(run_boundaries) - 1)]
    return runs


def pad_or_truncate_runs(run_grids: List[np.ndarray], target_steps: Optional[int] = None) -> List[np.ndarray]:
    if not run_grids:
        return run_grids

    step_counts = [grid.shape[0] for grid in run_grids]
    if target_steps is None:
        target_steps = max(step_counts)

    _, height, width = run_grids[0].shape
    padded_grids = []

    for i, grid in enumerate(run_grids):
        current_steps = grid.shape[0]
        if current_steps < target_steps:
            padding = np.zeros((target_steps - current_steps, height, width))
            padded_grids.append(np.concatenate([grid, padding], axis=0))
        else:
            padded_grids.append(grid[:target_steps])

    return padded_grids


def multiple_run_grid(csv_file_path: str,
                      value_column: str = 'income',
                      grid_size: Optional[Tuple[int, int]] = None,
                      pad_runs: bool = True) -> np.ndarray:
    df = pd.read_csv(csv_file_path)
    df[['x', 'y']] = df['pos'].apply(parse_position).apply(pd.Series)
    runs = detect_runs(df)

    height, width = determine_grid_size(df, grid_size)
    run_grids = []

    for i, (start_idx, end_idx) in enumerate(runs):
        run_df = df.iloc[start_idx:end_idx].copy()
        ts_grid = df_to_timeseries_grid(run_df, value_column, (height, width))
        run_grids.append(ts_grid)

    if pad_runs:
        run_grids = pad_or_truncate_runs(run_grids)
    else:
        step_counts = [grid.shape[0] for grid in run_grids]
        if len(set(step_counts)) > 1:
            raise ValueError("Runs have different step counts. Use pad_runs=True to standardize them.")

    print("Stacking runs into final 4D array...")
    result = np.stack(run_grids, axis=0)
    print(f"Final array shape: {result.shape}")
    return result
