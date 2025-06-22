import numpy as np

from src.income_distribution import load_distribution, plot_income_distribution
from src.model import GentSimModel
from src.simulation_runner import single_run

if __name__ == "__main__":

    single_run(10, 10, 10, 0.5, 1, 0.5, 10)
    
