import os
import pandas as pd
import numpy as np
from itertools import product
from SALib.sample import saltelli

from src.model import GentSimModel


def single_run(
    n_agents,
    n_neighborhoods,
    n_houses,
    epsilon,
    p_h,
    b,
    r_moore,
    sensitivity_param,
    steps,
    output_path="data/agent_data.csv",
    save_data=True,
):
    """
    Runs a single simulation of GentSimModel for a given number of steps.

    Parameters:
        n_agents (int): Number of agents.
        n_neighborhoods (int): Number of neighborhoods.
        n_houses (int): Number of houses.
        epsilon (float): Income-based decision threshold.
        p_h (float): Probability of household relocation attempt.
        sensitivity_param (float): Sensitivity parameter for income-based decisions.
        steps (int): Number of simulation steps to run.
        output_path (str): Path to save the agent-level data CSV.
        save_data (bool): Whether to save the DataFrame to CSV.
        verbose (bool): Whether to print progress and summary info.
        plot_income (bool): Whether to show the income distribution plot.
    """
    gentsim = GentSimModel(
        n_agents, n_neighborhoods, n_houses, epsilon, p_h, b, r_moore, sensitivity_param
    )

    for step in range(steps):
        print(f"Running step {step + 1}/{steps}...")
        gentsim.step()

    agent_df = gentsim.datacollector.get_agent_vars_dataframe()

    if save_data:
        agent_df.to_csv(output_path, index=True)
        print(f"Agent data saved to: {output_path}")

    return agent_df


def multiple_runs(
    n_agents,
    n_neighborhoods,
    n_houses,
    epsilon,
    p_h,
    b,
    r_moore,
    sensitivity_param,
    steps,
    runs=10,
    output_path="data/combined_agent_data.csv",
):
    """
    Runs multiple simulations of GentSimModel and saves the results.

    Parameters:
        n_agents (int): Number of agents.
        n_neighborhoods (int): Number of neighborhoods.
        n_houses (int): Number of houses.
        epsilon (float): Income-based decision threshold.
        p_h (float): Probability of household relocation attempt.
        sensitivity_param (int): Sensitivity parameter for income-based decisions.
        steps (int): Number of simulation steps to run.
        runs (int): Number of simulation runs to perform.
        output_path (str): Path to save the agent-level data CSV.
    """
    all_data = []

    for run in range(runs):
        print(f"Running simulation {run + 1}/{runs}...")
        agent_df = single_run(
            n_agents,
            n_neighborhoods,
            n_houses,
            epsilon,
            p_h,
            b,
            r_moore,
            sensitivity_param,
            steps,
            save_data=False,
        )
        all_data.append(agent_df)

    combined_df = pd.concat(all_data)
    combined_df.to_csv(output_path, index=True)
    print(f"Combined agent data saved to: {output_path}")


def parameter_sweep(n_agents, n_neighborhoods, n_houses, steps, runs, n_samples):
    """
    Performs a parameter sweep using SALib's Saltelli sampling.

    Parameters:
        n_agents (int): Number of agents.
        n_neighborhoods (int): Number of neighborhoods.
        n_houses (int): Number of houses.
        steps (int): Number of simulation steps.
        runs (int): Number of simulation runs per setting.
        n_samples (int): Number of base samples for Saltelli.
    """
    # Define the parameter space
    problem = {
        'num_vars': 5,
        'names': ['epsilon', 'p_h', 'b', 'r_moore', 'sensitivity_param'],
        'bounds': [
            [0, 10],       # epsilon (integer)
            [0.01, 0.3],   # p_h (probability)
            [0.0, 1.0],    # b (bounded float)
            [1, 3],        # r_moore (Moore radius, integer)
            [1, 10]        # sensitivity_param (weighing factor, integer)
        ]
    }

    # Generate parameter combinations using Saltelli sampling
    param_values = saltelli.sample(problem, n_samples, calc_second_order=False)

    # Convert appropriate parameters to integers
    param_values[:, 0] = np.round(param_values[:, 0])  # epsilon
    param_values[:, 3] = np.round(param_values[:, 3])  # r_moore
    param_values[:, 4] = np.round(param_values[:, 4])  # sensitivity_param

    # Ensure output directory exists
    output_dir = "data/sweep_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerated {len(param_values)} parameter sets using SALib.")

    # Run the simulations for each sampled parameter set
    for i, (epsilon, p_h, b, r_moore, sensitivity_param) in enumerate(param_values):
        print(f"\n=== Running SALib sweep {i + 1}/{len(param_values)} ===")

        filename = (
            f"agents_sweep{i}_eps{int(epsilon)}_ph{p_h:.2f}_b{b:.2f}_r{int(r_moore)}_s{int(sensitivity_param)}.csv"
        )
        output_path = os.path.join(output_dir, filename)

        multiple_runs(
            n_agents=n_agents,
            n_neighborhoods=n_neighborhoods,
            n_houses=n_houses,
            epsilon=int(epsilon),
            p_h=p_h,
            b=b,
            r_moore=int(r_moore),
            sensitivity_param=int(sensitivity_param),
            steps=steps,
            runs=runs,
            output_path=output_path,
        )

    print("\n✅ SALib parameter sweep completed.")
