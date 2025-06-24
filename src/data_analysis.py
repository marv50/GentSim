import numpy as np
from simulation_runner import multiple_runs


def average_across_runs(simulation_results):
    """
    Averages simulation results across runs.

    Parameters:
        simulation_results (np.ndarray): 4D array of shape (num_runs, num_steps, width, height)

    Returns:
        np.ndarray: 3D array of shape (num_steps, width, height) with averages across runs
    """
    return np.mean(simulation_results, axis=0)


