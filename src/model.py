from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from src.household import Household
from src.neighbourhood import Neighbourhood
import numpy as np


class GentSimModel(Model):
    """
    A model for simulating the GentSim environment.
    """

    def __init__(self, N: int, n: int, theta: float, epsilon: int, p_h: float) -> None:
        super().__init__()
        self.grid = SingleGrid(N * n, N * n, False)
        # self.num_agents = N * n
        self.N = N
        self.n = n
        self.theta = theta
        self.epsilon = epsilon
        self.p_h = p_h  # probability of high income households
        self.neighbourhoods = np.array(
            [[Neighbourhood(i, j) for i in range(N)] for j in range(N)],
            dtype=Neighbourhood,
        )

        self.datacollector = DataCollector(
            agent_reporters={
                "income": lambda a: a.income,
                "pos": lambda a: a.pos
            }
        )

        self.agent_lst = []  # list to hold all created agents
        self.grid_history = []

        self.empty_houses = np.ones((N * n, N * n), dtype=bool)
        self.init_population(N, n, 0.01)

    def init_population(self, N: int, n: int, p: float) -> None:
        """
        Initialize the population of agents in the model.
        """
        for i in range(N * n):
            for j in range(N * n):
                if self.random.random() < p:
                    agent = self.new_agent((i, j), N)
                    self.empty_houses[i, j] = False

    def new_agent(self, pos, N) -> None:
        """
        Create a new agent at the specified position.
        """
        household = Household(self, pos)
        neighbourhood = self.neighbourhoods[pos[0] // N, pos[1] // N]
        neighbourhood.residents += 1
        neighbourhood.total_income += household.income
        self.agent_lst.append(household)  # track the agent

        return household

    def get_current_income_grid(self) -> np.ndarray:
        """
        Create a snapshot of the current income distribution on the grid.

        Returns:
        - np.ndarray: 2D array where each cell contains the income of the agent at that position,
                      or 0 if the cell is empty.
        """
        income_grid = np.zeros((self.grid.width, self.grid.height))

        for agent in self.agent_lst:
            x, y = agent.pos
            income_grid[x, y] = agent.income

        return income_grid

    def save_grid_snapshot(self):
        """
        Save the current state of the grid to history.
        """
        current_grid = self.get_current_income_grid()
        self.grid_history.append(current_grid.copy())

        # Optionally limit history length to save memory
        # Keep only the last epsilon + 10 snapshots (some buffer)
        max_history_length = self.epsilon + 10
        if len(self.grid_history) > max_history_length:
            self.grid_history = self.grid_history[-max_history_length:]

    def step(self):
        """
        Advance the model by one step.
        """
        self.agents.shuffle_do("step", self)
        self.save_grid_snapshot()
        self.datacollector.collect(self)




gentsim = GentSimModel(10, 10, 0.5, 1, 0.5)
occupied_count = np.sum(~gentsim.empty_houses)
print(f"Total occupied houses: {occupied_count}")
for _ in range(10):  # Run for 10 steps

    gentsim.step()
    occupied_count = np.sum(~gentsim.empty_houses)
    print(f"Total occupied houses: {occupied_count}")


# Count occupied houses
occupied_count = np.sum(~gentsim.empty_houses)
print(f"Total occupied houses: {occupied_count}")

agent_df = gentsim.datacollector.get_agent_vars_dataframe()
print(agent_df.head(100))
