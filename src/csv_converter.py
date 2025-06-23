"""
csv_converter.py - Fixed Version

This module provides utilities to convert agent-based simulation data from a CSV file 
into a 3D or 4D NumPy array (time series of 2D spatial grids). Each grid slice represents 
the spatial distribution of a specified attribute (e.g., income or AgentID) at a particular time step.

Fixed issues:
- Improved parse_position function to handle np.int64() format more robustly
- Better error handling and debugging
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
        # First, try to extract numbers from np.int64() format
        numbers = re.findall(r'np\.int64\((\d+)\)', pos_str)
        if len(numbers) == 2:
            return int(numbers[0]), int(numbers[1])

        # If that doesn't work, try to extract any numbers from the string
        numbers = re.findall(r'\d+', pos_str)
        if len(numbers) == 2:
            return int(numbers[0]), int(numbers[1])

        # Last resort: try literal eval after cleaning the string
        # Remove np.int64() wrapper
        cleaned = re.sub(r'np\.int64\((\d+)\)', r'\1', pos_str)
        return tuple(ast.literal_eval(cleaned))

    except Exception as e:
        raise ValueError(f"Invalid position string: {pos_str}. Error: {e}")


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


def df_to_timeseries_grid(df: pd.DataFrame, value_column: str = 'income',
                          grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Convert a DataFrame of agent data into a 3D NumPy array representing
    a time series of spatial grids.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Step', 'pos', and value column.
        value_column (str): Name of the column to use for grid values.
        grid_size (tuple, optional): (height, width) of the grid. If None, inferred from data.

    Returns:
        np.ndarray: 3D array of shape (time_steps, height, width)
    """
    # Debug: print first few position strings
    print("Sample position strings:")
    print(df['pos'].head().tolist())

    # Parse positions with error handling
    try:
        df[['x', 'y']] = df['pos'].apply(parse_position).apply(pd.Series)
    except Exception as e:
        print(f"Error parsing positions: {e}")
        # Try alternative parsing

        def safe_parse(pos_str):
            try:
                return parse_position(pos_str)
            except:
                print(f"Failed to parse: {pos_str}")
                return (0, 0)  # Default fallback

        df[['x', 'y']] = df['pos'].apply(safe_parse).apply(pd.Series)

    height, width = determine_grid_size(df, grid_size)
    print(f"Grid size: {height} x {width}")

    steps = sorted(df['Step'].unique())
    print(f"Time steps: {steps}")

    ts_grid = np.zeros((len(steps), height, width))

    for i, step in enumerate(steps):
        step_data = df[df['Step'] == step]
        ts_grid[i] = create_grid_for_step(
            step_data, (height, width), value_column)

    return ts_grid


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
    return df_to_timeseries_grid(df, value_column, grid_size)


def multiple_run_grid(csv_file_path: str,
                      value_column: str = 'income',
                      grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Convert multiple runs of agent data from a CSV file into a 4D NumPy array.
    Detects new runs based on when the 'Step' column restarts at 1.

    Parameters:
        csv_file_path (str): Path to the input CSV file.
        value_column (str): Name of the column to use for grid values.
        grid_size (tuple, optional): (height, width) of the grid. If None, inferred from data.

    Returns:
        np.ndarray: 4D array with shape (runs, time_steps, height, width)
    """
    df = pd.read_csv(csv_file_path)

    # Debug: print data info
    print(f"Total records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("First few rows:")
    print(df.head())

    # Parse positions
    df[['x', 'y']] = df['pos'].apply(parse_position).apply(pd.Series)

    # Find the indices where 'Step' restarts to 1 — marking new runs
    run_start_indices = df.index[df['Step'] == 1].tolist()
    run_start_indices.append(len(df))  # Add sentinel to handle the final run

    print(f"Found {len(run_start_indices)-1} runs")
    print(f"Run start indices: {run_start_indices}")

    run_grids = []

    for i in range(len(run_start_indices) - 1):
        start_idx = run_start_indices[i]
        end_idx = run_start_indices[i + 1]
        run_df = df.iloc[start_idx:end_idx]

        print(f"Processing run {i + 1} with {len(run_df)} records")
        ts_grid = df_to_timeseries_grid(run_df, value_column, grid_size)
        run_grids.append(ts_grid)

    return np.stack(run_grids, axis=0)
