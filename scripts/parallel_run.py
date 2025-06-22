from src.model import GentSimModel
from mesa.batchrunner import batch_run

if __name__ == "__main__":

    params = {
        "N_agents": 50,
        "N_neighbourhoods": 100,
        "N_houses": 100,
        "theta": 0.5,
        "epsilon": 1,
        "p_h": 0.5
    }

    results = batch_run(
        GentSimModel,
        parameters=params, 
        number_processes=1)