from mesa import Model
from mesa.space import SingleGrid
from src.household import Household
from src.neighbourhood import Neighbourhood
import numpy as np


class GentSimModel(Model):
    """
    A model for simulating the GentSim environment.
    """

    def __init__(self, N: int, n: int, theta: float, epsilon: int, p_h: float) -> None:
        super().__init__()
        self.grid = SingleGrid(N*n, N*n, False)
        # self.num_agents = N * n
        self.N = N
        self.n = n
        self.theta = theta
        self.epsilon = epsilon
        self.p_h = p_h  # probability of high income households
        self.neighbourhoods = np.array([[Neighbourhood(i, j) for i in range(N)] for j in range(N)], dtype=Neighbourhood)
        self.empty_houses = np.zeros((N*n, N*n), dtype=bool)
        self.init_population(N, n, 0.5)
        self.income_history = {}  # needed for high income households


    def init_population(self, N: int, n: int, p: float) -> None:
        """ 
        Initialize the population of agents in the model.
        """
        for i in range(N*n):
            for j in range(N*n):
                if self.random.random() < p:
                    agent = self.new_agent((i, j), N)
                    self.empty_houses[i, j] = False
                

    def new_agent(self, pos, N) -> None:
        """
        Create a new agent at the specified position.
        """
        household = Household(self, pos)
        neighbourhood = self.neighbourhoods[pos[0]//N, pos[1]//N]
        neighbourhood.residents += 1
        neighbourhood.total_income += household.income

        return self.grid.place_agent(household, pos)

    def step(self):
        """
        Advance the model by one step.
        """
        self.agents.shuffle_do("step", self)

gentsim = GentSimModel(10, 10, 0.5)
