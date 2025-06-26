from src.simulation_runner import parameter_sweep
from src.csv_converter import multiple_run_grid
from src.data_analysis import analyze_sweep, spatial_income_disparity
import numpy as np
from matplotlib import pyplot as plt

# Definer number of parameters examined for the SA

if __name__ == "__main__":
    # Set the random seed for reproducibility
    n_agents = 300
    n_neighborhoods = 5
    n_houses = 5
    steps = 50
    runs = 10
    n_samples = 1

    parameter_sweep(
        n_agents,
        n_neighborhoods,
        n_houses,
        steps,
        runs,
        n_samples
    )

    results = analyze_sweep(spatial_income_disparity, n_neighborhoods, n_houses)
    print("Shape of results:", results.shape)
    print("Results:", results)
