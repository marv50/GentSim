import numpy as np

from src.model import GentSimModel
from src.simulation_runner import single_run
from src.csv_converter import *
from scripts.create_plots import plot_income_distribution, visualize_grid_evolution

if __name__ == "__main__":
    single_run(600, 5, 5, 0.5, 10, 0.8, 300)

    plot_income_distribution(
        title="Income Distribution in the Netherlands 2022",
        xlabel=r"Income $\times 1000$ (in Euros)",
        ylabel="Frequency",
    )

    file_path = 'data/agent_data.csv'

    # Convert using income
    income_grids = csv_to_timeseries_grid(file_path, value_column='income')
    print(f"Income grid shape: {income_grids.shape}")

    # Convert using agent ID
    agent_grids = csv_to_timeseries_grid(file_path, value_column='AgentID')
    print(f"Agent ID grid shape: {agent_grids.shape}")

    # Stats
    print(f"Grid size: {income_grids.shape[1]} x {income_grids.shape[2]}")
    print(f"Time steps: {income_grids.shape[0]}")
    print(f"Income range: {income_grids.min()} - {income_grids.max()}")

    # Plot
    visualize_grid_evolution(income_grids, save_path='fig/income_grid_evolution.png')

