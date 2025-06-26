import os
import pandas as pd
import numpy as np
import shutil
from itertools import product

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
            False  # save_data
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
    income_distribution=None,
    income_bounds=[1, 24.000, 71.200, 100.001],
):
    problem = {
        "num_vars": 5,
        "names": ["epsilon", "p_h", "b", "r_moore", "rent_factor"],
        "bounds": [
            [5, 10],       # epsilon
            [0.01, 0.3],   # p_h
            [0.0, 1.0],    # b
            [1, 2],        # r_moore
            [0.5, 0.9],    # rent_factor
        ],
    }

    param_values = saltelli.sample(problem, n_samples, calc_second_order=False)
    param_values[:, 0] = np.round(param_values[:, 0])  # epsilon
    param_values[:, 3] = np.round(param_values[:, 3])  # r_moore

    output_dir = "data/sweep_results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    total_runs = len(param_values)
    print(f"\nStarting SALib parameter sweep with {total_runs} parameter sets...\n")

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
            sensitivity_param=2,  # Hardcoded
            steps=steps,
            runs=runs,
            income_distribution=income_distribution,
            income_bounds=income_bounds,
            output_path=output_path,
        )

    print("\n✅ SALib parameter sweep completed.")



