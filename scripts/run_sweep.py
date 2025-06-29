from src.simulation_runner import parameter_sweep
from src.csv_converter import multiple_run_grid
from src.data_analysis import analyze_sweep, spatial_income_disparity
import numpy as np
from matplotlib import pyplot as plt

# Definer number of parameters examined for the SA

if __name__ == "__main__":
    n_agents = 300
    n_neighborhoods = 5
    n_houses = 5
    steps = 50
    runs = 10
    n_samples = 10

    problem = {
        "num_vars": 5,
        "names": ["epsilon", "p_h", "b", "r_moore", "rent_factor"],
        "bounds": [
            [2, 10],       # epsilon: agent tolerance
            [0.1, 0.9],   # p_h: probability of moving
            [0.0, 1.0],    # b: bias parameter
            [1, 2],        # r_moore: neighborhood radius
            [0.3, 0.7],    # rent_factor: rent multiplier
        ],
    }
    
    
    parameter_sweep(
        n_agents,
        n_neighborhoods,
        n_houses,
        steps,
        runs,
        n_samples,
        problem
    )

