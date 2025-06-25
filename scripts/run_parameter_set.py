from matplotlib import pyplot as plt

from src.simulation_runner import multiple_runs
from src.csv_converter import multiple_run_grid
from src.data_analysis import *

multiple_runs(
    n_agents=100,
    n_neighborhoods=5,
    n_houses=5,
    epsilon=5,
    p_h=0.2,
    b=0.5,
    r_moore=1,
    sensitivity_param=2,
    rent_factor=0.5,
    steps=50,
    runs=10,
    output_path='data/combined_agent_data.csv'
)


path = 'data/combined_agent_data.csv'
array = multiple_run_grid(path)  

result = spatial_income_disparity_over_time(array, N_neighbourhoods=5, N_houses=5)

plt.plot(result)
plt.xlabel('Time Step')
plt.ylabel('Average Spatial Income Disparity')
plt.title('Average Spatial Income Disparity Over Time')
plt.grid()
plt.savefig('fig/disparity_over_time.png')
