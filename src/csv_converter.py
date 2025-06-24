"""
csv_converter.py - Fixed Version with Enhanced Debugging

This module provides utilities to convert agent-based simulation data from a CSV file 
into a 3D or 4D NumPy array (time series of 2D spatial grids). Each grid slice represents 
the spatial distribution of a specified attribute (e.g., income or AgentID) at a particular time step.

Fixed issues:
- Improved parse_position function to handle np.int64() format more robustly
- Better error handling and debugging
- Fixed multiple_run_grid to handle runs with different numbers of time steps
"""

import numpy as np
import pandas as pd
import ast
import re
from typing import Tuple, Optional, List


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
                          grid_size: Optional[Tuple[int, int]] = None, debug: bool = False) -> np.ndarray:
    """
    Convert a DataFrame of agent data into a 3D NumPy array representing
    a time series of spatial grids.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Step', 'pos', and value column.
        value_column (str): Name of the column to use for grid values.
        grid_size (tuple, optional): (height, width) of the grid. If None, inferred from data.
        debug (bool): Whether to print debug information.

    Returns:
        np.ndarray: 3D array of shape (time_steps, height, width)
    """
    if debug:
        print("Sample position strings:")
        print(df['pos'].head().tolist())

    # Parse positions with error handling
    try:
        df = df.copy()  # Avoid modifying original DataFrame
        df[['x', 'y']] = df['pos'].apply(parse_position).apply(pd.Series)
    except Exception as e:
        if debug:
            print(f"Error parsing positions: {e}")
        # Try alternative parsing

        def safe_parse(pos_str):
            try:
                return parse_position(pos_str)
            except:
                if debug:
                    print(f"Failed to parse: {pos_str}")
                return (0, 0)  # Default fallback

        df[['x', 'y']] = df['pos'].apply(safe_parse).apply(pd.Series)

    height, width = determine_grid_size(df, grid_size)
    if debug:
        print(f"Grid size: {height} x {width}")

    steps = sorted(df['Step'].unique())
    if debug:
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


def detect_runs(df: pd.DataFrame, debug: bool = False) -> List[Tuple[int, int]]:
    """
    Detect run boundaries in the DataFrame based on Step column resets.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Step' column
        debug (bool): Whether to print debug information

    Returns:
        List[Tuple[int, int]]: List of (start_index, end_index) tuples for each run
    """
    # Find where Step decreases (indicating a new run)
    step_diff = df['Step'].diff()
    reset_indices = step_diff[step_diff < 0].index.tolist()

    # Create run boundaries
    run_boundaries = [0] + reset_indices + [len(df)]

    runs = []
    for i in range(len(run_boundaries) - 1):
        start_idx = run_boundaries[i]
        end_idx = run_boundaries[i + 1]
        runs.append((start_idx, end_idx))

    if debug:
        print(f"Detected {len(runs)} runs:")
        for i, (start, end) in enumerate(runs):
            run_steps = df.iloc[start:end]['Step'].unique()
            print(
                f"  Run {i+1}: indices {start}-{end}, steps {min(run_steps)}-{max(run_steps)} ({len(run_steps)} steps)")

    return runs


def pad_or_truncate_runs(run_grids: List[np.ndarray], target_steps: Optional[int] = None,
                         debug: bool = False) -> List[np.ndarray]:
    """
    Ensure all runs have the same number of time steps by padding with zeros or truncating.

    Parameters:
        run_grids (List[np.ndarray]): List of 3D arrays for each run
        target_steps (int, optional): Target number of time steps. If None, uses maximum.
        debug (bool): Whether to print debug information

    Returns:
        List[np.ndarray]: List of 3D arrays all with the same number of time steps
    """
    if not run_grids:
        return run_grids

    # Determine target number of steps
    step_counts = [grid.shape[0] for grid in run_grids]
    if target_steps is None:
        target_steps = max(step_counts)

    if debug:
        print(f"Run step counts: {step_counts}")
        print(f"Target steps: {target_steps}")

    # Get grid dimensions from first run
    _, height, width = run_grids[0].shape

    padded_grids = []
    for i, grid in enumerate(run_grids):
        current_steps = grid.shape[0]

        if current_steps == target_steps:
            padded_grids.append(grid)
        elif current_steps < target_steps:
            # Pad with zeros
            padding_shape = (target_steps - current_steps, height, width)
            padding = np.zeros(padding_shape)
            padded_grid = np.concatenate([grid, padding], axis=0)
            padded_grids.append(padded_grid)
            if debug:
                print(
                    f"Padded run {i+1} from {current_steps} to {target_steps} steps")
        else:
            # Truncate
            truncated_grid = grid[:target_steps]
            padded_grids.append(truncated_grid)
            if debug:
                print(
                    f"Truncated run {i+1} from {current_steps} to {target_steps} steps")

    return padded_grids


def multiple_run_grid(csv_file_path: str,
                      value_column: str = 'income',
                      grid_size: Optional[Tuple[int, int]] = None,
                      pad_runs: bool = True,  # New parameter
                      debug: bool = True) -> np.ndarray:
    """
    Convert multiple runs of agent data from a CSV file into a 4D NumPy array.
    Detects new runs based on resets in the 'Step' column.

    Parameters:
        csv_file_path (str): Path to the input CSV file.
        value_column (str): Name of the column to use for grid values.
        grid_size (tuple, optional): (height, width) of the grid. If None, inferred from data.
        pad_runs (bool): Whether to pad shorter runs with zeros to match longest run.
        debug (bool): Whether to print debug information.

    Returns:
        np.ndarray: 4D array with shape (runs, time_steps, height, width)
    """
    df = pd.read_csv(csv_file_path)

    if debug:
        print(f"Loaded CSV with {len(df)} rows")
        print(f"Step range: {df['Step'].min()} to {df['Step'].max()}")

    # Parse positions for the entire dataframe
    df[['x', 'y']] = df['pos'].apply(parse_position).apply(pd.Series)

    # Detect runs
    runs = detect_runs(df, debug=debug)

    # Determine grid size from entire dataset
    height, width = determine_grid_size(df, grid_size)
    if debug:
        print(f"Overall grid size: {height} x {width}")

    # Process each run
    run_grids = []
    for i, (start_idx, end_idx) in enumerate(runs):
        run_df = df.iloc[start_idx:end_idx].copy()

        if debug:
            print(f"\nProcessing run {i+1}:")
            print(f"  Data range: {start_idx} to {end_idx}")
            print(f"  Steps: {sorted(run_df['Step'].unique())}")

        # Convert this run to a 3D grid
        ts_grid = df_to_timeseries_grid(
            run_df, value_column, (height, width), debug=False)
        run_grids.append(ts_grid)

        if debug:
            print(f"  Generated grid shape: {ts_grid.shape}")

    # Handle different run lengths
    if pad_runs:
        run_grids = pad_or_truncate_runs(run_grids, debug=debug)
    else:
        # Check if all runs have same number of steps
        step_counts = [grid.shape[0] for grid in run_grids]
        if len(set(step_counts)) > 1:
            raise ValueError(f"Runs have different numbers of steps: {step_counts}. "
                             f"Set pad_runs=True to handle this automatically.")

    if debug:
        print(f"\nFinal run grid shapes: {[grid.shape for grid in run_grids]}")

    # Stack runs into a 4D array: (runs, time_steps, height, width)
    result = np.stack(run_grids, axis=0)

    if debug:
        print(f"Final 4D array shape: {result.shape}")

    return result
