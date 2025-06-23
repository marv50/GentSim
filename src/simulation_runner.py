import numpy as np
import pandas as pd

from src.model import GentSimModel


def single_run(
    n_agents,
    n_neighborhoods,
    n_houses,
    theta,
    epsilon,
    p_h,
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
        theta (float): Agent preference for neighborhood similarity.
        epsilon (float): Income-based decision threshold.
        p_h (float): Probability of household relocation attempt.
        steps (int): Number of simulation steps to run.
        output_path (str): Path to save the agent-level data CSV.
        save_data (bool): Whether to save the DataFrame to CSV.
        verbose (bool): Whether to print progress and summary info.
        plot_income (bool): Whether to show the income distribution plot.
    """
    gentsim = GentSimModel(n_agents, n_neighborhoods, n_houses, theta, epsilon, p_h)

    for step in range(steps):
        print(f"Running step {step + 1}/{steps}...")
        gentsim.step()

    agent_df = gentsim.datacollector.get_agent_vars_dataframe()

    if save_data:
        agent_df.to_csv(output_path, index=True)
        print(f"Agent data saved to: {output_path}")

    return agent_df
