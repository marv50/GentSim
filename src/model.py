import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid

from src.household import Household
from src.income_distribution import (
    create_income_distribution,
    load_distribution,
    custom_income_distribution,
)
from src.neighbourhood import Neighbourhood


class GentSimModel(Model):
    """
    A model for simulating the GentSim environment.
    """

    def __init__(
        self,
        N_agents: int = 10,
        N_neighbourhoods: int = 5,
        N_houses: int = 5,
        income_distribution: None | list = None,
        income_bounds: list = [1, 24_000, 71_200, 100_001],
        epsilon: int = 1,
        p_h: int = 0.5,
        b: float = 0.5,
        r_moore: int = 1,
        sensitivity_param: int = 2,
        rent_factor: float = 0.7,
    ) -> None:
        super().__init__()

        self.N_neighbourhoods = N_neighbourhoods
        self.N_houses = N_houses
        self.N_agents = N_agents
        self.rent_factor = (
            rent_factor  # Factor to calculate rent based on neighbourhood income
        )
        self.init_grid()

        self.epsilon = epsilon
        self.p_h = p_h
        self.b = b
        self.sensitivity_param = sensitivity_param
        self.r_moore = r_moore

        self.income_bounds = income_bounds
        assert isinstance(income_distribution, (list, type(None))), (
            f"Income distribution must be a list or None, not {type(income_distribution)}"
        )
        if income_distribution is None:
            income_distribution = create_income_distribution(
                load_distribution("data/income_data.csv")
            )
            self.income_samples = list(income_distribution.rvs(size=N_agents) * 1000)
        else:
            self.income_samples = list(
                custom_income_distribution(N_agents, income_distribution, income_bounds)
            )
        np.random.shuffle(self.income_samples)
        self.init_population(N_agents)

        self.init_datacollector()

    def init_grid(self) -> None:
        """
        Initialize the grid for the model.
        """
        assert self.N_neighbourhoods > 0, (
            "Number of neighbourhoods must be greater than 0"
        )
        assert self.N_houses > 0, "Number of houses must be greater than 0"

        self.grid = SingleGrid(
            self.N_neighbourhoods * self.N_houses,
            self.N_neighbourhoods * self.N_houses,
            False,
        )
        assert (
            self.N_agents
            <= self.N_neighbourhoods
            * self.N_neighbourhoods
            * self.N_houses
            * self.N_houses
        ), "Number of agents cannot exceed grid size"

        self.empty_houses = np.ones(
            (
                self.N_neighbourhoods * self.N_houses,
                self.N_neighbourhoods * self.N_houses,
            ),
            dtype=bool,
        )

        self.neighbourhoods = np.array(
            [
                [
                    Neighbourhood(self, i, j, self.rent_factor)
                    for i in range(self.N_neighbourhoods)
                ]
                for j in range(self.N_neighbourhoods)
            ],
            dtype=Neighbourhood,
        )

    def init_datacollector(self) -> None:
        """
        Initialize the DataCollector for the model.
        """
        self.datacollector = DataCollector(
            agent_reporters={"income": lambda a: a.income, "pos": lambda a: a.pos}
        )
        self.grid_history = []
        self.neighbourhood_history = []

    def init_population(self, N_agents: int) -> None:
        """
        Initialize the population of agents in the model.
        """
        for _ in range(N_agents):
            empty_houses = np.argwhere(self.empty_houses)
            if empty_houses.size == 0:
                return  # No empty houses left

            sample_pos = np.random.choice(empty_houses.shape[0])
            pos = tuple(empty_houses[sample_pos])
            assert isinstance(pos, tuple), "Position must be a tuple"

            agent = Household(self, income=self.income_samples.pop(0))
            self.update_neighbourhood(agent, pos)
            self.grid.place_agent(agent, pos)
            self.empty_houses[pos] = False  # Mark the house as occupied

    def update_neighbourhood(self, agent, pos) -> None:
        """
        Create a new agent at the specified position.
        """
        neighbourhood = self.neighbourhoods[
            pos[0] // self.N_neighbourhoods, pos[1] // self.N_neighbourhoods
        ]
        neighbourhood.residents += 1
        agent.neighbourhood = neighbourhood
        neighbourhood.total_income += agent.income
        # return

    def get_current_income_grid(self) -> np.ndarray:
        """
        Create a snapshot of the current income distribution on the grid.
        """
        income_grid = np.zeros((self.grid.width, self.grid.height))

        for agent in self.agents:
            x, y = agent.pos
            income_grid[x, y] = agent.income

        return income_grid

    def save_grid_snapshot(self):
        """
        Save the current state of the grid and neighborhood income to history.
        """
        current_grid = self.get_current_income_grid()
        self.grid_history.append(current_grid.copy())

        # Handle zero-resident divisions safely
        neighborhood_income = np.array(
            [
                [
                    (n.total_income / n.residents) if n.residents > 0 else 0.0
                    for n in row
                ]
                for row in self.neighbourhoods
            ]
        )
        self.neighbourhood_history.append(neighborhood_income)

        # Maintain bounded history length
        max_history_length = self.epsilon + 1
        if len(self.grid_history) > max_history_length:
            self.grid_history = self.grid_history[-max_history_length:]
        if len(self.neighbourhood_history) > max_history_length:
            self.neighbourhood_history = self.neighbourhood_history[
                -max_history_length:
            ]

    def step(self):
        """
        Advance the model by one step.
        """
        self.agents.shuffle_do("step", self)
        self.save_grid_snapshot()
        self.datacollector.collect(self)
