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


def spatial_income_disparity_over_time(income_data, N_neighbourhoods, N_houses):
    """
    Computes the average (max - min) neighborhood income at each timestep across all runs.

    Parameters:
        income_data (np.ndarray): 4D array of shape (n_runs, n_steps, width, height)
        N_neighbourhoods (int): Number of neighborhoods along one axis
        N_houses (int): Size of neighborhood block

    Returns:
        np.ndarray: 1D array of average max-min neighborhood income difference for each timestep
    """
    n_runs, n_steps, height, width = income_data.shape

    # Verify dimensions match expectations
    expected_size = N_neighbourhoods * N_houses
    if height != expected_size or width != expected_size:
        raise ValueError(f"Grid size ({height}x{width}) doesn't match "
                         f"expected {expected_size}x{expected_size}")

    disparity_over_time = []

    # Loop through each timestep
    for step in range(n_steps):
        # shape: (n_runs, width, height)
        step_frames = income_data[:, step, :, :]
        step_diffs = []

        # Loop through each run at this timestep
        for frame in step_frames:
            neighborhood_means = []

            # Calculate mean income for each neighborhood
            for i in range(N_neighbourhoods):
                for j in range(N_neighbourhoods):
                    start_i, end_i = i * N_houses, (i + 1) * N_houses
                    start_j, end_j = j * N_houses, (j + 1) * N_houses

                    block = frame[start_i:end_i, start_j:end_j]

                    if block.size == 0:
                        raise ValueError(
                            f"Empty block at neighborhood ({i}, {j}) at step {step}")

                    neighborhood_means.append(np.mean(block))

            if len(neighborhood_means) == 0:
                raise ValueError(f"No neighborhoods found at step {step}")

            # Calculate disparity for this run at this timestep
            disparity = max(neighborhood_means) - min(neighborhood_means)
            step_diffs.append(disparity)

        # Average disparity across all runs for this timestep
        disparity_over_time.append(np.mean(step_diffs))

    return np.array(disparity_over_time)


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
