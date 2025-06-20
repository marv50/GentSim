import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid

from household import Household
from neighbourhood import Neighbourhood
# from src.household import Household
# from src.neighbourhood import Neighbourhood


class GentSimModel(Model):
    """
    A model for simulating the GentSim environment.
    """

    def __init__(
        self,
        N_agents: int,
        N_neighbourhoods: int,
        N_houses: int,
        theta: float,
        epsilon: int,
        p_h: float,
        max_income: int = 100_000,
    ) -> None:
        super().__init__()

        self.N_neighbourhoods = N_neighbourhoods
        self.N_houses = N_houses
        self.N_agents = N_agents
        self.init_grid()

        self.theta = theta
        self.epsilon = epsilon
        self.p_h = p_h
        self.max_income = max_income
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
        assert self.N_agents <= self.N_neighbourhoods * self.N_neighbourhoods * self.N_houses * self.N_houses, (
            "Number of agents cannot exceed grid size"
        )

        self.empty_houses = np.ones(
            (
                self.N_neighbourhoods * self.N_houses,
                self.N_neighbourhoods * self.N_houses,
            ),
            dtype=bool,
        )

        self.neighbourhoods = np.array(
            [
                [Neighbourhood(i, j) for i in range(self.N_neighbourhoods)]
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
        self.agent_lst = []
        self.grid_history = []

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

            agent = Household(self, self.random.randint(1, self.max_income))
            self.grid.place_agent(agent, pos)
            self.empty_houses[pos] = False  # Mark the house as occupied

    def new_agent(self, pos, N) -> None:
        """
        Create a new agent at the specified position.
        """
        household = Household(self, pos)
        neighbourhood = self.neighbourhoods[pos[0] // N, pos[1] // N]
        neighbourhood.residents += 1
        neighbourhood.total_income += household.income
        self.agent_lst.append(household)  # track the agent

        self.agent_lst.append(household)

        return household

    def get_current_income_grid(self) -> np.ndarray:
        """
        Create a snapshot of the current income distribution on the grid.
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


if __name__ == "__main__":
    gentsim = GentSimModel(10, 10, 10, 0.5, 1, 0.5)
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
