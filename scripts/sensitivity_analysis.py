from SALib.analyze import morris
from SALib.sample.morris import sample

from src.csv_converter import multiple_run_grid
from src.data_analysis import analyze_sweep, average_income_final_step

import numpy as np
import matplotlib.pyplot as plt

# Define problem
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

# Sample input parameters using Morris
param_values = sample(problem, N=10, num_levels=4, optimal_trajectories=None)
# Run model (this assumes you already ran simulations and saved data)
y = analyze_sweep(average_income_final_step)

# Analyze
Si = morris.analyze(problem, param_values, y, conf_level=0.95, print_to_console=True, num_levels=4)

mu_star = Si["mu_star"]
sigma = Si["sigma"]
labels = problem["names"]

plt.figure(figsize=(8, 6))
plt.scatter(mu_star, sigma)

for i, label in enumerate(labels):
    plt.annotate(label, (mu_star[i], sigma[i]))

plt.xlabel(r"$\mu^*$ (Mean of absolute effects)")
plt.ylabel(r"$\sigma$ (Standard deviation)")
plt.title("Morris Sensitivity Analysis")
plt.grid(True)
plt.tight_layout()
plt.show()