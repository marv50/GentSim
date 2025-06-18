from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from src.model import GentSimModel
from src.household import Household

# Define how agents are drawn on the grid
def agent_portrayal(agent):
    if not isinstance(agent, Household):
        return

    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.5,
        "Layer": 1,
        "Color": "gray"
    }

    if agent.income < 1:  # adjust this based on how you define low/mid/high
        portrayal["Color"] = "red"
        portrayal["r"] = 0.3
    elif agent.income == 1:
        portrayal["Color"] = "green"
        portrayal["r"] = 0.4
    else:
        portrayal["Color"] = "blue"
        portrayal["r"] = 0.5

    return portrayal

# Grid dimensions
GRID_WIDTH = 10 * 10
GRID_HEIGHT = 10 * 10

grid = CanvasGrid(agent_portrayal, GRID_WIDTH, GRID_HEIGHT, 500, 500)

# Optional: add chart modules for visual metrics
# chart = ChartModule([{"Label": "Some Metric", "Color": "Black"}])

server = ModularServer(
    GentSimModel,
    [grid],  # Add chart here if needed: [grid, chart]
    "GentSim Model",
    {
        "N": UserSettableParameter("slider", "N (grid blocks)", 10, 1, 20, 1),
        "n": UserSettableParameter("slider", "n (block size)", 10, 1, 20, 1),
        "theta": UserSettableParameter("slider", "Theta", 0.5, 0.0, 1.0, 0.1),
        "epsilon": UserSettableParameter("slider", "Epsilon (growth window)", 5, 1, 50, 1),
        "p_h": UserSettableParameter("slider", "p_h (High-income move prob)", 0.1, 0.0, 1.0, 0.01)
    }
)

server.port = 8521  # Default port
server.launch()
