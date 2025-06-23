import numpy as np

from src.income_distribution import load_distribution, plot_income_distribution
from src.model import GentSimModel
from src.simulation_runner import single_run

if __name__ == "__main__":

    
    n_agents = 50
    n_neighborhoods = 5
    n_houses = 5
    theta = 0.5
    epsilon = 5
    p_h = 0.8
    steps=50
    single_run(n_agents, n_neighborhoods, n_houses, theta, epsilon, p_h, steps)
    
