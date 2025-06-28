import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.csv_converter import multiple_run_grid


def average_income_at_step(data, step):
    """
    Calculate the average income at a specific time step.

    Parameters:
        data (np.ndarray): 4D array with shape (runs, steps, width, height).
        step (int): The time step to evaluate.

    Returns:
        float: The average income at the given step.
    """
    return np.mean(data[:, step, :, :])


def average_income_over_time(data):
    """
    Calculate the average income at each time step over all runs and spatial dimensions.

    Parameters:
        data (np.ndarray): 4D array with shape (runs, steps, width, height).

    Returns:
        np.ndarray: 1D array of average income for each time step.
    """
    steps = data.shape[1]
    return np.array([average_income_at_step(data, step) for step in range(steps)])


def average_income_final_step(data):
    """
    Calculate the average income at the final time step.

    Parameters:
        data (np.ndarray): 4D array with shape (runs, steps, width, height).

    Returns:
        float: The average income at the final step.
    """
    final_step = data.shape[1] - 1
    return average_income_at_step(data, final_step)


def spatial_income_disparity(income_data, N_neighbourhoods, N_houses):
    """
    Computes the average (max - min) neighborhood income at final timestep across all runs.
    """
    final_frames = income_data[:, -1, :, :]  # shape: (n_runs, width, height)
    height, width = final_frames.shape[1], final_frames.shape[2]

    # Verify dimensions match expectations
    expected_size = N_neighbourhoods * N_houses
    if height != expected_size or width != expected_size:
        raise ValueError(f"Grid size ({height}x{width}) doesn't match "
                         f"expected {expected_size}x{expected_size}")

    diffs = []
    for frame in final_frames:
        neighborhood_means = []
        for i in range(N_neighbourhoods):
            for j in range(N_neighbourhoods):
                start_i, end_i = i * N_houses, (i + 1) * N_houses
                start_j, end_j = j * N_houses, (j + 1) * N_houses

                block = frame[start_i:end_i, start_j:end_j]

                if block.size == 0:
                    raise ValueError(f"Empty block at neighborhood ({i}, {j})")

                neighborhood_means.append(np.mean(block))

        if len(neighborhood_means) == 0:
            raise ValueError("No neighborhoods found")

        disparity = max(neighborhood_means) - min(neighborhood_means)
        diffs.append(disparity)

    return np.mean(diffs)


def spatial_income_disparity_over_time(income_data, N_neighbourhoods, N_houses, return_sem=True):
    """
    Computes the average (max - min) neighborhood income at each timestep across all runs,
    and optionally returns the standard deviation or standard error.

    Parameters:
        income_data (np.ndarray): Shape (n_runs, n_steps, width, height)
        N_neighbourhoods (int): Number of neighborhoods per axis
        N_houses (int): Number of houses per neighborhood side
        return_sem (bool): If True, return standard error; else return standard deviation

    Returns:
        mean_disparities (np.ndarray): Mean disparity per timestep (shape: n_steps)
        uncertainty (np.ndarray): Std or SEM per timestep (shape: n_steps)
    """
    n_runs, n_steps, height, width = income_data.shape
    expected_size = N_neighbourhoods * N_houses
    if height != expected_size or width != expected_size:
        raise ValueError(f"Grid size ({height}x{width}) doesn't match expected {expected_size}x{expected_size}")

    mean_disparities = []
    uncertainties = []

    for step in range(n_steps):
        step_frames = income_data[:, step, :, :]
        step_diffs = []

        for frame in step_frames:
            neighborhood_means = []
            for i in range(N_neighbourhoods):
                for j in range(N_neighbourhoods):
                    start_i, end_i = i * N_houses, (i + 1) * N_houses
                    start_j, end_j = j * N_houses, (j + 1) * N_houses
                    block = frame[start_i:end_i, start_j:end_j]
                    if block.size == 0:
                        raise ValueError(f"Empty block at neighborhood ({i}, {j}) at step {step}")
                    neighborhood_means.append(np.mean(block))

            disparity = max(neighborhood_means) - min(neighborhood_means)
            step_diffs.append(disparity)

        step_diffs = np.array(step_diffs)
        mean_disparities.append(np.mean(step_diffs))
        if return_sem:
            uncertainties.append(np.std(step_diffs, ddof=1) / np.sqrt(len(step_diffs)))
        else:
            uncertainties.append(np.std(step_diffs, ddof=1))

    return np.array(mean_disparities), np.array(uncertainties)


def analyze_sweep(metric, *args, **kwargs):
    """
    Analyzes simulation results from multiple CSV files in correct numerical order,
    applying the given metric to each.
    """
    directory = "data/sweep_results"
    files = glob.glob(os.path.join(directory, "*.csv"))

    # Sort numerically by extracting the number from the filename
    def extract_index(filepath):
        match = re.search(r'parameter_sweep_(\d+)\.csv', os.path.basename(filepath))
        return int(match.group(1)) if match else float('inf')

    files_sorted = sorted(files, key=extract_index)

    results = []

    for file_path in files_sorted:
        print(f"Processing file: {file_path}")

        simulation_data = multiple_run_grid(file_path)
        result = metric(simulation_data, *args, **kwargs)
        results.append(result)

    return np.array(results)



