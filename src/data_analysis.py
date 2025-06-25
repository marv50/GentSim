import os
import glob
import numpy as np

from src.csv_converter import multiple_run_grid


def average_across_runs(simulation_results):
    """
    Averages simulation results across runs.

    Parameters:
        simulation_results (np.ndarray): 4D array of shape (num_runs, num_steps, width, height)

    Returns:
        np.ndarray: 3D array of shape (num_steps, width, height) with averages across runs
    """
    return np.mean(simulation_results, axis=0)


def spatial_income_disparity(income_data, N_neighbourhoods, N_houses):
    """
    Computes the average (max - min) neighborhood income at final timestep across all runs.

    Parameters:
        income_data (np.ndarray): 4D array of shape (n_runs, n_steps, width, height)
        N_neighbourhoods (int): Number of neighborhoods along one axis
        N_houses (int): Size of neighborhood block
    Returns:
        float: average max-min neighborhood income difference at final step
    """
    final_frames = income_data[:, -1, :, :]  # shape: (n_runs, width, height)
    diffs = []

    for frame in final_frames:
        neighborhood_means = []
        for i in range(N_neighbourhoods):
            for j in range(N_neighbourhoods):
                block = frame[
                    i * N_houses : (i + 1) * N_houses, j * N_houses : (j + 1) * N_houses
                ]
                block_mean = np.mean(block)
                neighborhood_means.append(block_mean)
        disparity = max(neighborhood_means) - min(neighborhood_means)
        diffs.append(disparity)


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

    # return np.mean(diffs)
