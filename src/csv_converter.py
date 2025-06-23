"""
csv_converter.py

This module provides utilities to convert agent-based simulation data from a CSV file 
into a 3D NumPy array (time series of 2D spatial grids). Each grid slice represents 
the spatial distribution of a specified attribute (e.g., income or AgentID) at a particular time step.

Typical use case:
- Input: CSV file with columns 'Step', 'pos' (position tuple), and an attribute column like 'income'.
- Output: 3D array of shape (time_steps, grid_height, grid_width).

Functions:
- parse_position: Parse position string into (x, y) coordinates.
- determine_grid_size: Infer grid size from agent positions or use a specified one.
- create_grid_for_step: Convert data for a single time step into a 2D grid.
- csv_to_timeseries_grid: Main interface to convert CSV data into a 3D time series grid.
"""

import numpy as np
import pandas as pd
import ast
import re
from typing import Tuple, Optional


def parse_position(pos_str: str) -> Tuple[int, int]:
    """
    Parse a string representing a position tuple into (x, y) integers.
    
    Supports formats like:
        "(np.int64(20), np.int64(15))"
        "(20, 15)"

    Parameters:
        pos_str (str): The position string.

    Returns:
        Tuple[int, int]: Parsed (x, y) position.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        numbers = re.findall(r'np\.int64\((\d+)\)', pos_str)
        if len(numbers) == 2:
            return int(numbers[0]), int(numbers[1])
        return tuple(ast.literal_eval(pos_str))
    except Exception:
        raise ValueError(f"Invalid position string: {pos_str}")


def determine_grid_size(df: pd.DataFrame, grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Determine the grid dimensions (height, width) for the spatial layout.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'x' and 'y' columns.
        grid_size (tuple, optional): If provided, overrides inferred size.

    Returns:
        Tuple[int, int]: (grid_height, grid_width)
    """
    if grid_size:
        return grid_size
    return df['y'].max() + 1, df['x'].max() + 1


def create_grid_for_step(step_data: pd.DataFrame, grid_shape: Tuple[int, int], value_column: str) -> np.ndarray:
    """
    Create a 2D grid representing agent values at a specific time step.

    Parameters:
        step_data (pd.DataFrame): Subset of the data for one time step.
        grid_shape (tuple): Shape of the grid (height, width).
        value_column (str): Column to extract grid values from (e.g., 'income').

    Returns:
        np.ndarray: 2D array of shape (height, width) with values placed on the grid.
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
    Convert a CSV file of agent data into a 3D NumPy array representing
    a time series of spatial grids.

    Parameters:
        csv_file_path (str): Path to the input CSV file.
        value_column (str): Name of the column to use for grid values.
        grid_size (tuple, optional): (height, width) of the grid. If None, inferred from data.

    Returns:
        np.ndarray: 3D array with shape (time_steps, height, width)
    """
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
