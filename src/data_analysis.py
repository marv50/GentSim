import os
import glob
import numpy as np

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


def analyze_sweep(metric, *args, **kwargs):
    """
    Analyzes simulation results from multiple CSV files and applies a metric.
    """

    directory = "data/sweep_results"
    files = glob.glob(os.path.join(directory, "*.csv"))

    results = []

    for file_path in files:
        print(f"Processing file: {file_path}")

        # Pass the file path to your conversion function
        simulation_data = multiple_run_grid(file_path)
        result = metric(simulation_data, *args, **kwargs)
        results.append(result)

    return np.array(results)
