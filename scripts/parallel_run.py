from mesa.batchrunner import batch_run

from src.model import GentSimModel

if __name__ == "__main__":
    params = {
        "N_agents": 50,
        "N_neighbourhoods": 100,
        "N_houses": 100,
        "epsilon": 1,
        "p_h": 0.5,
        "b": 0.5,
        "r_moore": 1,
        "sensitivity_param": 2,
    }

    results = batch_run(GentSimModel, parameters=params, number_processes=1)
