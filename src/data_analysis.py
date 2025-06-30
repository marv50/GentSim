import os
import glob
import re
import numpy as np

from src.csv_converter import multiple_run_grid


def average_income_at_step(data: np.ndarray, step: int) -> float:
    """
    Calculate the average income at a specific time step.

    Parameters:
    - data (np.ndarray): 4D array with shape (n_runs, n_steps, width, height).
    - step (int): The time step to evaluate.

    Returns:
    - float: The average income at the given step.
    """
    return np.mean(data[:, step, :, :])


def average_income_over_time(data: np.ndarray) -> np.ndarray:
    """
    Calculate the average income at each time step over all runs and spatial dimensions.

    Parameters:
    - data (np.ndarray): 4D array with shape (n_runs, n_steps, width, height).

    Returns:
    - np.ndarray: 1D array of average income for each time step.
    """
    steps = data.shape[1]
    return np.array([average_income_at_step(data, step) for step in range(steps)])


def average_income_final_step(data: np.ndarray) -> float:
    """
    Calculate the average income at the final time step.

    Parameters:
    - data (np.ndarray): 4D array with shape (n_runs, n_steps, width, height).

    Returns:
    - float: The average income at the final step.
    """
    final_step = data.shape[1] - 1
    return average_income_at_step(data, final_step)


def clustering_at_step(
    data: np.ndarray,
    step: int,
    N_neighbourhoods: int = 5,
    N_houses: int = 5,
    bins: list = [1, 24_000, 71_200, 100_001],
):
    """
    Calculate the clustering coefficient at a specific time step.

    Parameters:
    - data (np.ndarray): 4D array with shape (n_runs, n_steps, width, height).
    - step (int): The time step to evaluate.
    - N_neighbourhoods (int): Number of neighborhoods per axis.
    - N_houses (int): Number of houses per neighborhood side.
    - bins (int): Number of bins for clustering calculation.

    Returns:
    - float: The average clustering coefficient at the given step.
    """
    # Assuming clustering is defined as the average of all values in the grid
    clustering_values = []
    frame = data[:, step, :, :]  # shape: (n_runs, width, height)
    for run in frame:
        clustering_at_step = []
        for i in range(N_neighbourhoods):
            for j in range(N_neighbourhoods):
                start_i, end_i = i * N_houses, (i + 1) * N_houses
                start_j, end_j = j * N_houses, (j + 1) * N_houses
                block = run[start_i:end_i, start_j:end_j]

                if block.size == 0:
                    print(block)
                    print(run.shape)
                    print(run)
                    raise ValueError(f"Empty block at neighborhood ({i}, {j})")
                # Count poor houses in the block
                # Assuming bins[1] is the threshold for poor income
                l = np.sum((block >= bins[0]) & (block < bins[1]))
                m = np.sum((block >= bins[1]) & (
                    block < bins[2]))  # Mid-income houses
                h = np.sum(block >= bins[2])  # Rich houses
                clustering = (
                    (l * l + m * m + h * h) /
                    (l + m + h) if (l + m + h) > 0 else 1
                )
                clustering_at_step.append(clustering)

        clustering_values.append(np.mean(clustering_at_step))
    return clustering_values


def average_clustering_over_time(
    data: np.ndarray,
    N_neighbourhoods: int = 5,
    N_houses: int = 5,
    bins: list = [1, 24_000, 71_200, 100_001],
) -> tuple:
    """
    Calculate the average clustering coefficient over time.

    For each time step, computes the average clustering across all runs.

    Parameters:
    - data (np.ndarray): 4D array with shape (n_runs, n_steps, width, height).
    - N_neighbourhoods (int): Number of neighborhoods per axis.
    - N_houses (int): Number of houses per neighborhood side.
    - bins (int): Number of bins for clustering calculation.

    Returns:
    - np.ndarray: 1D array of average clustering coefficients for each time step.
    """
    n_steps = data.shape[1]
    avg_clustering = np.array(
        [
            clustering_at_step(data, step, N_neighbourhoods, N_houses, bins)
            for step in range(n_steps)
        ]
    )
    mean_avg_clustering = np.mean(
        avg_clustering, axis=1)  # Average across runs
    
    std = np.std(avg_clustering, axis=1, ddof=1)
    return mean_avg_clustering, std


def clustering_scalar(
    data: np.ndarray,
    N_neighbourhoods: int = 5,
    N_houses: int = 5,
    bins: list = [1, 24_000, 71_200, 100_001],
) -> float:
    """
    Calculate the average clustering coefficient across all runs and time steps.

    Parameters:
    - data (np.ndarray): 4D array with shape (n_runs, n_steps, width, height).
    - N_neighbourhoods (int): Number of neighborhoods per axis.
    - N_houses (int): Number of houses per neighborhood side.
    - bins (list): Bins for clustering calculation.

    Returns:
    - float: The average clustering coefficient across all runs and time steps.
    """
    mean_avg_clustering, _ = average_clustering_over_time(
        data, N_neighbourhoods, N_houses, bins
    )
    return np.mean(mean_avg_clustering)


def spatial_income_disparity(
    income_data: np.ndarray, N_neighbourhoods: int, N_houses: int
) -> float:
    """
    Compute the average income disparity across neighborhoods at the final timestep.

    Disparity is defined as the average difference between the richest and poorest
    neighborhood incomes across all runs.

    Parameters:
    - income_data (np.ndarray): 4D array (n_runs, n_steps, width, height).
    - N_neighbourhoods (int): Number of neighborhoods per axis.
    - N_houses (int): Number of houses per neighborhood side.

    Returns:
    - float: Mean disparity at the final timestep across runs.
    """
    final_frames = income_data[:, -1, :, :]  # shape: (n_runs, width, height)
    height, width = final_frames.shape[1], final_frames.shape[2]

    expected_size = N_neighbourhoods * N_houses
    if height != expected_size or width != expected_size:
        raise ValueError(
            f"Grid size ({height}x{width}) doesn't match "
            f"expected {expected_size}x{expected_size}"
        )

    disparities = []
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
        disparities.append(disparity)

    return np.mean(disparities)


def spatial_income_disparity_over_time(
    income_data: np.ndarray,
    N_neighbourhoods: int,
    N_houses: int,
    return_sem: bool = False,
):
    """
    Compute the average neighborhood income disparity (max - min) over time.

    For each timestep, calculates the disparity across neighborhoods averaged over all runs.
    Optionally returns uncertainty as standard error or standard deviation.

    Parameters:
    - income_data (np.ndarray): 4D array (n_runs, n_steps, width, height).
    - N_neighbourhoods (int): Number of neighborhoods per axis.
    - N_houses (int): Number of houses per neighborhood side.
    - return_sem (bool): If True, return standard error; else return standard deviation.

    Returns:
    - mean_disparities (np.ndarray): Mean disparity per timestep (shape: n_steps).
    - uncertainty (np.ndarray): Uncertainty (SEM or std) per timestep (shape: n_steps).
    """
    n_runs, n_steps, height, width = income_data.shape
    expected_size = N_neighbourhoods * N_houses
    if height != expected_size or width != expected_size:
        raise ValueError(
            f"Grid size ({height}x{width}) doesn't match expected {expected_size}x{expected_size}"
        )

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
                        raise ValueError(
                            f"Empty block at neighborhood ({i}, {j}) at step {step}"
                        )

                    neighborhood_means.append(np.mean(block))

            disparity = max(neighborhood_means) - min(neighborhood_means)
            step_diffs.append(disparity)

        step_diffs = np.array(step_diffs)
        mean_disparities.append(np.mean(step_diffs))
        if return_sem:
            uncertainties.append(
                np.std(step_diffs, ddof=1) / np.sqrt(len(step_diffs)))
        else:
            uncertainties.append(np.std(step_diffs, ddof=1))

    return np.array(mean_disparities), np.array(uncertainties)


def analyze_sweep(metric: callable, *args, **kwargs) -> np.ndarray:
    """
    Analyze simulation results from a parameter sweep by applying a given metric
    function to each sweep result file in the correct numerical order.

    Parameters:
    - metric (callable): Function to apply to the data from each sweep result.
    - *args, **kwargs: Additional arguments forwarded to the metric function.

    Returns:
    - np.ndarray: Array of metric values, one for each parameter sweep file.
    """
    directory = "data/sweep_results"
    files = glob.glob(os.path.join(directory, "*.csv"))

    # Sort numerically using indices in filenames like parameter_sweep_X.csv
    def extract_index(filepath):
        match = re.search(r"parameter_sweep_(\d+)\.csv",
                          os.path.basename(filepath))
        return int(match.group(1)) if match else float("inf")

    files_sorted = sorted(files, key=extract_index)

    results = []

    for file_path in files_sorted:
        print(f"Processing file: {file_path}")
        simulation_data = multiple_run_grid(file_path)
        result = metric(simulation_data, *args, **kwargs)
        results.append(result)

    return np.array(results)
