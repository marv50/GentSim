"""
simulation_runner.py

This module provides functions for running single or multiple simulations
of the GentSim agent-based model, and performing parameter sweeps using
Morris sampling via SALib for sensitivity analysis.

Key functions:
- single_run: Runs the model once and returns/saves the agent-level data.
- multiple_runs: Runs the model multiple times in parallel and combines results.
- parameter_sweep: Conducts a parameter sweep over sampled parameter sets.

Author: [Your Name]
Date: [Date]
"""

import os
import pandas as pd
import numpy as np
import shutil
from SALib.sample import saltelli
from src.model import GentSimModel
from concurrent.futures import ProcessPoolExecutor


def single_run(
    n_agents,
    n_neighborhoods,
    n_houses,
    rent_factor,
    epsilon,
    p_h,
    b,
    r_moore,
    sensitivity_param,
    steps,
    income_distribution=None,
    income_bounds=[1, 24_000, 71_200, 100_001],
    output_path="data/agent_data.csv",
    save_data=True,
):
    """
    Run a single instance of the GentSimModel for a given number of steps.

    Parameters:
        n_agents (int): Number of agents in the model.
        n_neighborhoods (int): Number of neighborhoods.
        n_houses (int): Number of houses.
        rent_factor (float): Rent price multiplier.
        epsilon (int): Tolerance threshold.
        p_h (float): Probability of moving.
        b (float): Bias parameter.
        r_moore (int): Neighborhood radius.
        sensitivity_param (float): Sensitivity parameter (fixed or swept).
        steps (int): Number of simulation steps.
        income_distribution (str, optional): Income distribution specification.
        income_bounds (list, optional): Income group boundaries.
        output_path (str, optional): Where to save the agent data CSV.
        save_data (bool, optional): If True, writes agent data to disk.

    Returns:
        pandas.DataFrame: DataFrame containing agent-level simulation data.
    """
    gentsim = GentSimModel(
        N_agents=n_agents,
        N_neighbourhoods=n_neighborhoods,
        N_houses=n_houses,
        income_distribution=income_distribution,
        income_bounds=income_bounds,
        epsilon=epsilon,
        p_h=p_h,
        b=b,
        r_moore=r_moore,
        sensitivity_param=sensitivity_param,
        rent_factor=rent_factor,
    )

    for step in range(steps):
        if (step % 10 == 0) or step == 0:
            print(f"Running step {step + 1}/{steps}...")
        gentsim.step()

    agent_df = gentsim.datacollector.get_agent_vars_dataframe()

    if save_data:
        agent_df.to_csv(output_path, index=True)
        print(f"Agent data saved to: {output_path}")

    return agent_df


def _single_run_wrapper(args):
    """
    Wrapper for single_run to allow parallel execution with ProcessPoolExecutor.
    """
    return single_run(*args)


def multiple_runs(
    n_agents,
    n_neighborhoods,
    n_houses,
    rent_factor,
    epsilon,
    p_h,
    b,
    r_moore,
    sensitivity_param,
    steps,
    runs,
    income_distribution=None,
    income_bounds=[1, 24_000, 71_200, 100_001],
    output_path="data/combined_agent_data.csv",
):
    """
    Run multiple independent simulations of GentSimModel in parallel,
    combining the agent-level outputs into a single CSV.

    Parameters:
        (Same as single_run) plus:
        runs (int): Number of independent runs to execute.
        output_path (str): Filepath to save combined data.

    Returns:
        None. Combined data is saved to disk.
    """
    run_args = [
        (
            n_agents,
            n_neighborhoods,
            n_houses,
            rent_factor,
            epsilon,
            p_h,
            b,
            r_moore,
            sensitivity_param,
            steps,
            income_distribution,
            income_bounds,
            output_path,
            False  # Don't save individual runs
        )
        for _ in range(runs)
    ]

    print(f"Running {runs} simulations in parallel...")

    with ProcessPoolExecutor() as executor:
        all_data = list(executor.map(_single_run_wrapper, run_args))

    combined_df = pd.concat(all_data)
    combined_df.to_csv(output_path, index=True)
    print(f"Combined agent data saved to: {output_path}")


def parameter_sweep(
    n_agents,
    n_neighborhoods,
    n_houses,
    steps,
    runs,
    n_samples,
    problem,
    n_levels=4,
    income_distribution=None,
    income_bounds=[1, 24_000, 71_200, 100_001],
):
    """
    Perform a parameter sweep using Morris sampling (SALib) over specified
    parameter ranges. Each sampled parameter set triggers multiple model runs.

    Parameters:
        n_agents (int): Number of agents.
        n_neighborhoods (int): Number of neighborhoods.
        n_houses (int): Number of houses.
        steps (int): Steps per simulation.
        runs (int): Repeats per parameter set.
        n_samples (int): Number of parameter sets to sample.
        n_levels (int, optional): Discretization levels in Morris method.
        income_distribution (str, optional): Income distribution specification.
        income_bounds (list, optional): Income group boundaries.

    Returns:
        None. Saves results for each parameter set to disk.
    """

    param_values = saltelli.sample(problem, n_samples, calc_second_order=False)
    param_values[:, 0] = np.round(param_values[:, 0])  # epsilon to integer
    param_values[:, 3] = np.round(param_values[:, 3])  # r_moore to integer

    output_dir = "data/sweep_results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    total_runs = len(param_values)
    print(
        f"\nStarting SALib parameter sweep with {total_runs} parameter sets...\n")

    for i, (epsilon, p_h, b, r_moore, rent_factor) in enumerate(param_values):
        print(f"=== Running SALib sweep {i + 1} of {total_runs} ===")

        filename = f"parameter_sweep_{i + 1}.csv"
        output_path = os.path.join(output_dir, filename)

        multiple_runs(
            n_agents=n_agents,
            n_neighborhoods=n_neighborhoods,
            n_houses=n_houses,
            rent_factor=rent_factor,
            epsilon=int(epsilon),
            p_h=p_h,
            b=b,
            r_moore=int(r_moore),
            sensitivity_param=2,
            steps=steps,
            runs=runs,
            income_distribution=income_distribution,
            income_bounds=income_bounds,
            output_path=output_path,
        )

    print("Parameter sweep completed.")
