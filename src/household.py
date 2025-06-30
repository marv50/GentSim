import numpy as np
from numba import njit
from mesa import Agent, Model


class Household(Agent):
    """
    A household agent in the GentSim model.
    """

    def __init__(self, model: Model, income: int) -> None:
        """
        Initialize the household agent with a position and income.
        The position is set to the model's empty houses, and the income is assigned.

        Parameters:
        - model (Model): The model instance.
        - income (int): The income of the household.
        """
        super().__init__(model)
        self.income = income
        self.income_bin = get_income_bin(income, model.income_bounds)
        self.neighbourhood = None

    def step(self, model: Model) -> None:
        """
        We movin or not
        The household checks if it can afford the rent in its current neighbourhood.

        Parameters:
        - model (Model): The model instance.
        """
        neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in self.pos)
        ]

        if self.income < neighbourhood.rent():
            # If the household cannot afford the rent, it moves out
            if self.income_bin == "low":
                new_location = self.move_in(
                    model,
                    utility_func=self.move_in_low,
                )
                if new_location == 0:
                    # self.kill(model)
                    self.replace(model)
                    return
            elif self.income_bin == "medium":
                new_location = self.move_in(
                    model,
                    utility_func=self.move_in_medium,
                )
                if new_location == 0:
                    # self.kill(model)
                    self.replace(model)
                    return
            elif self.income_bin == "high":
                new_location = self.move_in(
                    model,
                    utility_func=self.move_in_high,
                )
                if new_location == 0:
                    # self.kill(model)
                    self.replace(model)
                    return
            if not new_location:
                return
            self.move(model, new_location)

        if self.income_bin == "low":
            move_out_prob = self.move_out_low(model, self.pos)
            if model.random.random() >= move_out_prob:
                return
            new_location = self.move_in(
                model,
                utility_func=self.move_in_low,
            )
            if not new_location:
                return
            self.move(model, new_location)

        if self.income_bin == "medium":
            move_out_prob = self.move_out_medium(model, self.pos)
            if model.random.random() >= move_out_prob:
                return
            new_location = self.move_in(model, utility_func=self.move_in_medium)
            if not new_location:
                return
            self.move(model, new_location)

        if self.income_bin == "high":
            if model.random.random() >= model.p_h:
                return
            new_location = self.move_in(
                model,
                utility_func=self.move_in_high,
            )
            if not new_location:
                return
            self.move(model, new_location)

    def kill(self, model: Model) -> None:
        """
        Remove the household from the model.

        Parameters:
        - model (Model): The model instance.
        """
        model.empty_houses[self.pos] = True
        neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in self.pos)
        ]
        neighbourhood.residents -= 1
        neighbourhood.total_income -= self.income
        assert bool(model.empty_houses[self.pos]) is True, "Position must be empty"
        assert neighbourhood.residents >= 0, "Residents must be non-negative"
        assert neighbourhood.total_income >= 0, "Total income must be non-negative"
        model.grid.remove_agent(self)
        self.remove()

    def replace(self, model: Model) -> None:
        """
        Replace the household with a new one in the same neighbourhood.

        Parameters:
        - model (Model): The model instance.
        """
        neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in self.pos)
        ]
        new_income = int(neighbourhood.rent() * 1.5)
        neighbourhood.total_income += new_income - self.income
        self.income = new_income
        self.income_bin = get_income_bin(new_income, model.income_bounds)

    def move(self, model: Model, location: tuple) -> None:
        """
        Move the household to a new location.

        Parameters:
        - model (Model): The model instance.
        - location (tuple): The new position to move to.
        """
        old_pos = self.pos
        model.empty_houses[old_pos] = True
        model.empty_houses[location] = False

        assert bool(model.empty_houses[self.pos]) is True or old_pos == location, (
            f"Old position ({old_pos}) ({self.pos}), must be empty ({model.empty_houses[self.pos]}) or equal to new position ({location})"
        )
        assert bool(model.empty_houses[location]) is False, (
            "New position must not be empty"
        )

        # Update the neighbourhoods
        old_neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in old_pos)
        ]
        new_neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in location)
        ]

        old_neighbourhood.residents -= 1
        new_neighbourhood.residents += 1
        new_neighbourhood.total_income += self.income
        old_neighbourhood.total_income -= self.income

        self.neighbourhood = new_neighbourhood

        model.grid.move_agent(self, location)

    def income_percentile(self, model: Model, target: tuple) -> float:
        """
        Calculate the income percentile of the household.

        Parameters:
        - model (Model): The model instance.
        - target (tuple): The target position to calculate the income percentile for.

        Returns:
        - float: The income percentile of the household in the local neighbourhood and chunk.
        """
        assert self.income > 0, "Income must be greater than 0"

        local_neighbours = model.grid.get_neighbors(
            target, moore=True, include_center=False, radius=model.r_moore
        )
        local_total = sum([n.income for n in local_neighbours])
        local_ip = self.income / (local_total + self.income)

        chunk_total = model.neighbourhoods[
            target[0] // model.N_neighbourhoods, target[1] // model.N_neighbourhoods
        ].total_income

        if (
            target[0] // model.N_neighbourhoods == self.pos[0] // model.N_neighbourhoods
        ) and (
            target[1] // model.N_neighbourhoods == self.pos[1] // model.N_neighbourhoods
        ):
            chunk_ip = self.income / chunk_total
        else:
            chunk_ip = self.income / (chunk_total + self.income)

        b = model.b
        ip = b * chunk_ip + (1 - b) * local_ip
        assert 0 <= ip <= 1, (
            f"Income percentile must be between 0 and 1 but got {ip}: b={b}, chunk_ip={chunk_ip}, local_ip={local_ip}, income={self.income}, chunk_total={chunk_total}, local_total={local_total}"
        )
        return ip

    def move_out_low(self, model: Model, pos: tuple) -> float:
        """
        Calculate the probability of moving out based on the income percentile.

        Parameters:
        - model (Model): The model instance.
        - pos (tuple): The position of the household.

        Returns:
        - float: The probability of moving out, which is a function of the income percentile.
        """
        gamma = self.income_percentile(model, pos)
        p = 1 - gamma ** (1 / model.sensitivity_param)
        assert 0 <= p <= 1
        return p

    def move_out_medium(self, model: Model, pos: tuple) -> float:
        """
        Calculate the probability of moving out based on the income percentile.

        Parameters:
        - model (Model): The model instance.
        - pos (tuple): The position of the household.

        Returns:
        - float: The probability of moving out, which is a function of the income percentile.
        """
        p = 4 * (self.income_percentile(model, pos) - 0.5) ** model.sensitivity_param
        assert 0 <= p <= 1
        return p

    def move_in_low(self, model: Model, pos: tuple) -> float:
        """
        Calculate the probability of moving in based on the income percentile.

        Parameters:
        - model (Model): The model instance.
        - pos (tuple): The position of the household.

        Returns:
        - float: The probability of moving in, which is the inverse of the move out probability
        """
        neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in pos)
        ]

        if self.income < neighbourhood.rent():
            # If the household cannot afford the rent, it cannot move in
            return 0.0
        p = 1 - self.move_out_low(model, pos)
        assert 0 <= p <= 1
        return p

    def move_in_medium(self, model: Model, pos: tuple) -> float:
        """
        Calculate the probability of moving in based on the income percentile.

        Parameters:
        - model (Model): The model instance.
        - pos (tuple): The position of the household.

        Returns:
        - float: The probability of moving in, which is the inverse of the move out probability
        """
        neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in pos)
        ]

        if self.income < neighbourhood.rent():
            # If the household cannot afford the rent, it cannot move in
            return 0.0
        p = 1 - self.move_out_medium(model, pos)
        return p

    def move_in_high(self, model: Model, pos: tuple) -> float:
        """
        Compute average income growth rate phi^epsilon(t) for a cell, required for high
        income households to move in somewhere else.

        Parameters:
        - model (Model): The model instance.
        - pos (tuple): The position of the household.

        Returns:
        - float: The average income growth rate for the household to move in.
        """
        if len(model.grid_history) < model.epsilon + 1:
            return 0.0

        nhood_x, nhood_y = (
            pos[0] // model.N_neighbourhoods,
            pos[1] // model.N_neighbourhoods,
        )
        neighbourhood = model.neighbourhoods[(nhood_x, nhood_y)]

        if self.income < neighbourhood.rent():
            return 0.0

        # Get recent grids and stack into one 3D NumPy array: (T, X, Y)
        recent_grids = model.grid_history[-(model.epsilon + 1) :]
        stacked_grids = np.stack(recent_grids)  # shape: (epsilon+1, width, height)

        # Get neighbor coordinates as arrays
        neighbor_positions = np.array(
            model.grid.get_neighborhood(
                pos, moore=True, include_center=False, radius=model.r_moore
            )
        )
        xs, ys = neighbor_positions[:, 0], neighbor_positions[:, 1]

        # Fast median computation
        medians = compute_neighbor_medians(stacked_grids, xs, ys)
        diffs_local = np.diff(medians)
        avg_growth_local = np.mean(diffs_local) if len(diffs_local) > 0 else 0.0

        # Global neighborhood differences
        x_idx, y_idx = nhood_x, nhood_y
        recent_neighbourhoods = model.neighbourhood_history[-(model.epsilon + 1) :]
        global_values = [nh[x_idx, y_idx] for nh in recent_neighbourhoods]
        diffs_global = np.diff(global_values)
        avg_growth_global = np.mean(diffs_global) if len(diffs_global) > 0 else 0.0

        return model.b * avg_growth_global + (1 - model.b) * avg_growth_local

    def move_in(self, model: Model, utility_func: callable, **kwargs) -> tuple:
        """
        Calculate the utility of moving into a house.
        The utility is calculated as the inverse of the utility function.
        Where the utility function depends on the income level

        Parameters:
        - model (Model): The model instance.
        - utility_func (callable): The utility function to use for calculating the utility.
        - **kwargs: Additional keyword arguments to pass to the utility function.

        Returns:
        - tuple: The position of the house to move into, or None if no suitable house is found.
        """
        empty_indices = np.argwhere(model.empty_houses)

        house_utilities = {
            tuple(idx): utility_func(model, **kwargs, pos=tuple(idx))
            for idx in empty_indices
        }
        total_sum = sum(house_utilities.values())
        if total_sum == 0:
            return 0  # All utilities are 0

        houses_probs = {
            new_pos: value / (total_sum) if (total_sum - value) != 0 else 0
            for new_pos, value in house_utilities.items()
        }

        items = list(houses_probs.items())
        np.random.shuffle(items)  # Randomize order
        for new_pos, prob in items:
            if prob >= np.random.rand():
                return new_pos
        return None


def get_income_bin(income: float, bins: list) -> str:
    """
    Get the income bin for the given income.

    Parameters:
    - income (float): The income of the household.
    - bins (list): A list of income bounds, where:
        - bins[0] is the lower bound for low income,
        - bins[1] is the upper bound for low income,
        - bins[2] is the upper bound for medium income,
        - bins[3] is the upper bound for high income.

    Returns:
    - str: The income bin, which can be "low", "medium", or "high".
    """
    assert isinstance(bins, list) and len(bins) == 4, (
        "bins must be a list of three elements"
    )
    low_income = bins[1]
    medium_income = bins[2]

    if income < low_income:
        return "low"
    elif low_income <= income <= medium_income:
        return "medium"
    elif income > medium_income:
        return "high"
    else:
        raise ValueError("Income must be greater than 0")


@njit
def compute_neighbor_medians(
    recent_grids: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """
    Compute the median values of neighbors in recent grids.

    Parameters:
    - recent_grids (np.ndarray): 3D array of shape (T, X, Y) containing recent grid values.
    - xs (np.ndarray): 1D array of x-coordinates of neighbors.
    - ys (np.ndarray): 1D array of y-coordinates of neighbors.

    Returns:
    - np.ndarray: 1D array of median values for each time step.
    """
    T = recent_grids.shape[0]
    n = xs.shape[0]
    medians = np.empty(T)

    for t in range(T):
        values = []
        for i in range(n):
            val = recent_grids[t, xs[i], ys[i]]
            if val > 0:
                values.append(val)
        if len(values) > 0:
            medians[t] = np.median(np.array(values))
        else:
            medians[t] = 0.0
    return medians
