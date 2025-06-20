import numpy as np

from scripts.income_distribution import load_distribution, plot_income_distribution
from src.model import GentSimModel

if __name__ == "__main__":
    gentsim = GentSimModel(10, 10, 10, 0.5, 1, 0.5)
    occupied_count = np.sum(~gentsim.empty_houses)
    print(f"Total occupied houses: {occupied_count}")
    for _ in range(10):  # Run for 10 steps
        gentsim.step()
        occupied_count = np.sum(~gentsim.empty_houses)
        print(f"Total occupied houses: {occupied_count}")

    # Count occupied houses
    occupied_count = np.sum(~gentsim.empty_houses)
    print(f"Total occupied houses: {occupied_count}")

    agent_df = gentsim.datacollector.get_agent_vars_dataframe()
    print(agent_df.head(100))

    income_df = load_distribution("data/income_data.csv")
    plot_income_distribution(
        income_df,
        title="Income Distribution in the Netherlands 2022",
        xlabel=r"Income $\times 1000$",
        ylabel="Frequency",
    )
