from src.simulation_runner import multiple_runs
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

multiple_runs(
    n_agents,
    n_neighborhoods,
    n_houses,
    steps,
    runs,
    n_samples
    )

path = "data/combined_agent_data.csv"

array = multiple_run_grid(path)

results = analyze_sweep(spatial_income_disparity, n_neighborhoods, n_houses)

plt.plot
