from src.simulation_runner import parameter_sweep
from src.csv_converter import multiple_run_grid
from src.data_analysis import analyze_sweep, spatial_income_disparity
import numpy as np
from matplotlib import pyplot as plt

n_agents = 50
n_neighborhoods = 5
n_houses = 5
steps = 10
runs = 2
n_samples = 2

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


