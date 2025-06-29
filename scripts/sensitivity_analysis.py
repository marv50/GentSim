from SALib.analyze import sobol

from src.csv_converter import multiple_run_grid
from src.data_analysis import analyze_sweep, average_income_final_step

import numpy as np
import matplotlib.pyplot as plt

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

# Sample input parameters using Morris
# Run model (this assumes you already ran simulations and saved data)
y = analyze_sweep(average_income_final_step)

# Analyze
Si = sobol.analyze(problem, y)

