import numpy as np
from mesa import Agent, Model


class Household(Agent):
    """
    A household agent in the GentSim model.
    """

    def __init__(self, model: Model, income: int) -> None:
        super().__init__(model)
        self.income = income
        self.income_bin = get_income_bin(income, model.income_bounds)

    def step(self, model: Model) -> None:
        """
        We movin or not
        """
        neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in self.pos)
        ]

        if self.income < neighbourhood.rent():
            # print(f"Household at {self.pos} with income {self.income} cannot afford rent {neighbourhood.rent()}.")
            # print(f"Total income in neighbourhood: {neighbourhood.total_income}, residents: {neighbourhood.residents}")
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

    def kill(self, model):
        """
        Remove the household from the model.
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
        print(f"Household at {self.pos} with income {self.income} has been removed.")
        model.grid.remove_agent(self)
        self.remove()

    def replace(self, model):
        neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in self.pos)
        ]
        new_income = int(neighbourhood.rent() * 1.5)
        neighbourhood.total_income += new_income - self.income
        self.income = new_income
        self.income_bin = get_income_bin(new_income, model.income_bounds)
        print(
            f"Household at {self.pos} has been replaced with new income {self.income}."
        )

    def move(self, model, location):
        """
        Move the household to a new location.
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

        model.grid.move_agent(self, location)

    def income_percentile(self, model, target) -> float:
        """
        Calculate the income percentile of the household.
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

    def move_out_low(self, model, pos) -> float:
        """
        Calculate the probability of moving out based on the income percentile.
        """
        gamma = self.income_percentile(model, pos)
        p = 1 - gamma ** (1 / model.sensitivity_param)
        assert 0 <= p <= 1
        return p

    def move_out_medium(self, model, pos):
        """
        Calculate the probability of moving out based on the income percentile.
        """
        p = 4 * (self.income_percentile(model, pos) - 0.5) ** model.sensitivity_param
        assert 0 <= p <= 1
        return p

    def move_in_low(self, model, pos) -> float:
        """
        Calculate the probability of moving in based on the income percentile.
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

    def move_in_medium(self, model, pos) -> float:
        """
        Calculate the probability of moving in based on the income percentile.
        """
        neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in pos)
        ]

        if self.income < neighbourhood.rent():
            # If the household cannot afford the rent, it cannot move in
            return 0.0
        p = 1 - self.move_out_medium(model, pos)
        # assert 0 <= p <= np.sqrt(gamma)
        return p

    def move_in_high(self, model, pos) -> float:
        """
        Compute average income growth rate phi^epsilon(t) for a cell, required for high
        income households to move in somewhere else.
        """
        if len(model.grid_history) < model.epsilon + 1:
            # print("Not enough history for high income to calculate phi^epsilon(t)")
            return 0.0

        neighbourhood = model.neighbourhoods[
            tuple(ti // model.N_neighbourhoods for ti in pos)
        ]
        if self.income < neighbourhood.rent():
            # If the household cannot afford the rent, it cannot move in
            return 0.0

        recent_grids = model.grid_history[-(model.epsilon + 1) :]

        neighbor_positions = model.grid.get_neighborhood(
            pos, moore=True, include_center=False, radius=model.r_moore
        )

        medians = []
        for grid_snapshot in recent_grids:
            neighbor_incomes = []
            for neighbor_pos in neighbor_positions:
                x, y = neighbor_pos
                income = grid_snapshot[x, y]
                if income > 0:  # Only include occupied cells
                    neighbor_incomes.append(income)

            if neighbor_incomes:
                medians.append(np.median(neighbor_incomes))
            else:
                medians.append(0.0)

        # Calculate the sum of differences over epsilon periods
        if len(medians) < model.epsilon + 1:
            print("return 0.0")
            return 0.0
        # Calculate differences: [t-(epsilon-1)] - [t-epsilon], [t-(epsilon-2)] - [t-(epsilon-1)], ..., [t] - [t-1]
        diffs = []
        for i in range(model.epsilon):
            diff = medians[i + 1] - medians[i]  # newer - older
            diffs.append(diff)

        average_growth_rate_local = sum(diffs) / model.epsilon

        recent_neighbourhoods = model.neighbourhood_history[-(model.epsilon + 1) :]

        diffs_global = []
        for i in range(model.epsilon):
            diff_global = (
                recent_neighbourhoods[i + 1][
                    pos[0] // model.N_neighbourhoods, pos[1] // model.N_neighbourhoods
                ]
                - recent_neighbourhoods[i][
                    pos[0] // model.N_neighbourhoods, pos[1] // model.N_neighbourhoods
                ]
            )
            diffs_global.append(diff_global)
        average_growth_rate_global = sum(diffs_global) / model.epsilon
        return (
            model.b * average_growth_rate_global
            + (1 - model.b) * average_growth_rate_local
        )

    def move_in(self, model, utility_func, **kwargs) -> tuple:
        """
        Calculate the utility of moving into a house.
        The utility is calculated as the inverse of the utility function.
        Where the utility function depends on the income level
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


def get_income_bin(income: float, bins: list) -> str:  # Fixed return type annotation
    """
    Get the income bin for the given income.
    """
    assert isinstance(bins, list) and len(bins) == 4, (
        "bins must be a list of three elements"
    )
    low_income = bins[1]
    medium_income = bins[2]

    if income < low_income:
        return "low"
    elif low_income <= income <= medium_income:  # Fixed condition to handle edge case
        return "medium"
    elif income > medium_income:  # Fixed condition to handle edge case
        return "high"
    else:
        raise ValueError("Income must be greater than 0")
