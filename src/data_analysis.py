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


if __name__ == "__main__":
    df = multiple_runs(
        n_agents=300,
        n_neighborhoods=5,
        n_houses=5,
        epsilon=5,
        p_h=0.2,
        b=0.5,
        r_moore=1,
        sensitivity_param=2,
        steps=10,
        runs=2,
    )

path = "data/combined_agent_data.csv"
array = multiple_run_grid(path, value_column='income')
print(array.shape)  # Should print (runs, time_steps, height, width)
