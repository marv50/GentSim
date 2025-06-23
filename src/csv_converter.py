import numpy as np
import pandas as pd
import ast
import re
from typing import Tuple, Optional


def parse_position(pos_str: str) -> Tuple[int, int]:
    try:
        numbers = re.findall(r'np\.int64\((\d+)\)', pos_str)
        if len(numbers) == 2:
            return int(numbers[0]), int(numbers[1])
        return tuple(ast.literal_eval(pos_str))
    except Exception:
        raise ValueError(f"Invalid position string: {pos_str}")


def determine_grid_size(df: pd.DataFrame, grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if grid_size:
        return grid_size
    return df['y'].max() + 1, df['x'].max() + 1


def create_grid_for_step(step_data: pd.DataFrame, grid_shape: Tuple[int, int], value_column: str) -> np.ndarray:
    grid = np.zeros(grid_shape)
    for _, row in step_data.iterrows():
        x, y = int(row['x']), int(row['y'])
        if 0 <= y < grid_shape[0] and 0 <= x < grid_shape[1]:
            grid[y, x] = row[value_column]
    return grid


def csv_to_timeseries_grid(csv_file_path: str, value_column: str = 'income',
                           grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    df = pd.read_csv(csv_file_path)
    df[['x', 'y']] = df['pos'].apply(parse_position).apply(pd.Series)
    height, width = determine_grid_size(df, grid_size)
    shape = (len(df['Step'].unique()), height, width)

    steps = sorted(df['Step'].unique())
    ts_grid = np.zeros(shape)

    for i, step in enumerate(steps):
        step_data = df[df['Step'] == step]
        ts_grid[i] = create_grid_for_step(step_data, (height, width), value_column)

    return ts_grid
