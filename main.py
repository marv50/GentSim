import numpy as np

from src.model import GentSimModel
from src.simulation_runner import single_run
from scripts.create_plots import plot_income_distribution

if __name__ == "__main__":
<<<<<<< HEAD

    
    n_agents = 50
    n_neighborhoods = 5
    n_houses = 5
    theta = 0.5
    epsilon = 5
    p_h = 0.8
    steps=50
    single_run(n_agents, n_neighborhoods, n_houses, theta, epsilon, p_h, steps)
    
=======
    single_run(10, 10, 10, 0.5, 1, 0.5, 10)

    plot_income_distribution(
        title="Income Distribution in the Netherlands 2022",
        xlabel=r"Income $\times 1000$ (in Euros)",
        ylabel="Frequency",
    )
>>>>>>> 4d1c4cb3d53937e58389e837fe77828af515ea52
