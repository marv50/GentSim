import numpy as np
from mesa import Model
from mesa.space import SingleGrid

from household import Household
from neighbourhood import Neighbourhood


class GentSimModel(Model):
    """
    A model for simulating the GentSim environment.
    """

    def __init__(
        self,
        N_agents: int,
        m: int,
        n: int,
        theta: float,
        epsilon: int,
        p_h: float,
        max_income: int = 100_000,
    ) -> None:
        super().__init__()

        self.grid = SingleGrid(m, n, False)
        self.m = m
        self.n = n
        assert N_agents <= m * n, "Number of agents cannot exceed grid size"

        self.empty_houses = np.ones((m, n), dtype=bool)
        self.max_income = max_income
        self.init_population(N_agents)

        self.theta = theta
        self.epsilon = epsilon
        self.p_h = p_h

        self.neighbourhoods = np.array(
            [[Neighbourhood(i, j) for i in range(m)] for j in range(m)],
            dtype=Neighbourhood,
        )
        self.income_history = {}  # needed for high income households

    def init_population(self, N_agents: int) -> None:
        """
        Randomly place N_agents in the grid of size m x n.

        ## Parameters
        - N_agents: Number of agents to place.

        ## Returns
        - None
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

        return household

    def step(self):
        """
        Advance the model by one step.
        """
        self.agents.shuffle_do("step", self)


if __name__ == "__main__":
    gentsim = GentSimModel(4, 5, 5, 0.5, 1, 0.5)
    occupied_count = np.sum(~gentsim.empty_houses)
    print(f"Total occupied houses: {occupied_count}")
    for _ in range(10):  # Run for 10 steps
        gentsim.step()
        occupied_count = np.sum(~gentsim.empty_houses)
        print(f"Total occupied houses: {occupied_count}")

    # Count occupied houses
    occupied_count = np.sum(~gentsim.empty_houses)
    print(f"Total occupied houses: {occupied_count}")
