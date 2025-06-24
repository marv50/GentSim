import os
import glob
import numpy as np

from simulation_runner import multiple_runs
from csv_converter import multiple_run_grid


def average_across_runs(simulation_results):
    """
    Averages simulation results across runs.

    Parameters:
        simulation_results (np.ndarray): 4D array of shape (num_runs, num_steps, width, height)

    Returns:
        np.ndarray: 3D array of shape (num_steps, width, height) with averages across runs
    """
    return np.mean(simulation_results, axis=0)


def analyze_sweep(metric, *args, **kwargs):

    directory = 'data/sweep_results'
    files = glob.glob(os.path.join(directory, '*.csv'))

    results = []

    for file_path in files:
        print(f"Processing file: {file_path}")

        # Pass the file path to your conversion function
        simulation_data = multiple_run_grid(file_path)
        result = metric(simulation_data, *args, **kwargs)
        results.append(result)
    
    return np.array(results)