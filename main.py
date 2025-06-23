import numpy as np

from src.model import GentSimModel
from src.simulation_runner import single_run
from scripts.create_plots import plot_income_distribution

if __name__ == "__main__":
    single_run(10, 10, 10, 0.5, 1, 0.5, 10)

    plot_income_distribution(
        title="Income Distribution in the Netherlands 2022",
        xlabel=r"Income $\times 1000$ (in Euros)",
        ylabel="Frequency",
    )
