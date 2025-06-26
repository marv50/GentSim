from matplotlib import pyplot as plt

from src.simulation_runner import multiple_runs
from src.csv_converter import multiple_run_grid
from src.data_analysis import *
from scripts.create_plots import plot_spatial_income_disparity_over_time

if __name__ == "__main__":
    multiple_runs(
        n_agents=300,
        n_neighborhoods=5,
        n_houses=5,
        epsilon=5,
        p_h=0.6,
        b=0.5,
        r_moore=1,
        sensitivity_param=2,
        rent_factor=0.9,
        steps=50,
        runs=10,
        output_path='data/combined_agent_data.csv'
    )


    path = 'data/combined_agent_data.csv'
    array = multiple_run_grid(path)  

    result = spatial_income_disparity_over_time(array, N_neighbourhoods=5, N_houses=5)

    plot_spatial_income_disparity_over_time(result)
