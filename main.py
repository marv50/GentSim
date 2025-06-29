from scripts.create_plots import plot_income_distribution, visualize_grid_evolution
from src.csv_converter import *
from src.simulation_runner import single_run

if __name__ == "__main__":
    bins = [1, 24_000, 71_200, 100_001]
    n_houses = 5

    single_run(
        n_agents=400,
        n_neighborhoods=5,
        n_houses=n_houses,
        epsilon=8,
        p_h=0.5,
        b=0.5,
        r_moore=1,
        sensitivity_param=2,
        steps=50,
        rent_factor=0.7,
        income_bounds=bins,
    )

    plot_income_distribution(
        title="Income Distribution in the Netherlands 2022",
        xlabel=r"Income $\times 1000$ (in Euros)",
        ylabel="Frequency",
    )

    file_path = "data/agent_data.csv"

    # Convert using income
    income_grids = csv_to_timeseries_grid(file_path, value_column="income")
    print(f"Income grid shape: {income_grids.shape}")

    # Convert using agent ID
    agent_grids = csv_to_timeseries_grid(file_path, value_column="AgentID")
    print(f"Agent ID grid shape: {agent_grids.shape}")

    # Stats
    print(f"Grid size: {income_grids.shape[1]} x {income_grids.shape[2]}")
    print(f"Time steps: {income_grids.shape[0]}")
    print(f"Income range: {income_grids.min()} - {income_grids.max()}")

    # Plot
    visualize_grid_evolution(
        income_grids,
        n_houses=n_houses,
        income_bounds=bins,
        save_path="fig/income_grid_evolution.png",
    )
