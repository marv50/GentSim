from SALib.analyze import sobol
from src.simulation_runner import parameter_sweep
from src.csv_converter import multiple_run_grid
from src.data_analysis import analyze_sweep, spatial_income_disparity, average_income_final_step
import numpy as np
from matplotlib import pyplot as plt

# Definer number of parameters examined for the SA

n_agents = 300
n_neighborhoods = 5
n_houses = 5
steps = 50
runs = 10
n_samples = 100

parameter_sweep(
    n_agents,
    n_neighborhoods,
    n_houses,
    steps,
    runs,
    n_samples
)
y = analyze_sweep(average_income_final_step)

problem = {
        "num_vars": 5,
        "names": ["epsilon", "p_h", "b", "r_moore", "rent_factor"],
        "bounds": [
            [0, 10],       # epsilon
            [0.01, 0.1],   # p_h
            [0.1, 1.0],    # b
            [1, 3],        # r_moore
            [0.5, 0.9],    # rent_factor
        ],
    }

Si = sobol.analyze(
    problem,
    y,
    print_to_console=True,
    calc_second_order=False)