import numpy as np
from mesa import Agent, Model


class Household(Agent):
    """
    A household agent in the GentSim model.
    """

    def __init__(self, model: Model, income: int) -> None:
        super().__init__(model)
        self.income = income
        self.income_bin = get_income_bin(income)
        # self.pos = pos
        # self.ph

    def step(self, model: Model) -> None:
        """
        We movin or not
        """
        if self.income_bin == "low":
            move_out_prob = move_out_low(model, self.income, self.pos)
            if model.random.random() < move_out_prob:
                new_location = move_in(
                    model,
                    utility_func=move_in_low,
                    income=self.income,
                )
                if new_location:
                    self.move(model, new_location)

        if self.income_bin == "medium":
            move_out_prob = move_out_medium(model, self.income, self.pos)
            if model.random.random() < move_out_prob:
                new_location = move_in(
                    model,
                    utility_func=move_in_medium,
                    income=self.income,
                )
                if new_location:
                    self.move(model, new_location)

        # if self.income_bin == "high":
        #     if model.random.random() < model.p_h:
        #         new_location = move_in(
        #             model,
        #             utility_func=move_in_high,
        #         )
        #         if new_location:
        #             self.move(model, new_location)

    def move(self, model, location):
        """
        Move the household to a new location.
        """
        model.empty_houses[self.pos] = True
        model.empty_houses[location] = False

        assert bool(model.empty_houses[self.pos]) is True, "Old position must be empty"
        assert bool(model.empty_houses[location]) is False, (
            "New position must not be empty"
        )

        model.grid.move_agent(self, location)


def income_percentile(model, income, pos) -> float:
    """
    Calculate the income percentile of the household.
    """
    assert income > 0, "Income must be greater than 0"
    neighbours = model.grid.get_neighbors(pos, True, False)
    total = sum([n.income for n in neighbours])
    return income / (total + income)


def move_out_low(model, income, pos) -> float:
    """
    Calculate the probability of moving out based on the income percentile.
    """
    gamma = income_percentile(model, income, pos)
    p = 1 - np.sqrt(gamma)
    assert 0 <= p <= 1
    return p


def move_out_medium(model, income, pos):
    """
    Calculate the probability of moving out based on the income percentile.
    """
    p = 4 * (income_percentile(model, income, pos) - 0.5) ** 2
    assert 0 <= p <= 1
    return p


def move_in_low(model, income, pos) -> float:
    """
    Calculate the probability of moving in based on the income percentile.
    """
    gamma = income_percentile(model, income, pos)
    p = np.sqrt(gamma)
    assert 0 <= p <= 1
    return p


def move_in_medium(model, income, pos) -> float:
    """
    Calculate the probability of moving in based on the income percentile.
    """
    p = 1 - move_out_medium(model, income, pos)
    assert 0 <= p <= 1
    return p


def move_in_high(model, pos) -> float:
    """
    Compute average income growth rate phi^epsilon(t) for a cell, required for h
    to move in somewhere else.
    """
    history = model.income_history[pos]
    if len(history) < model.epsilon + 1:
        return 0.0
    diffs = [history[-(i + 1)] - history[-(i + 2)] for i in range(model.epsilon)]
    return sum(diffs) / model.epsilon


def move_in(model, utility_func, **kwargs) -> tuple:
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


def get_income_bin(income: float) -> int:
    """
    Get the income bin for the given income.
    """
    if income < 38690:
        return "low"
    elif 38690 < income < 77280:
        return "medium"
    elif income > 77280:
        return "high"
    else:
        raise ValueError("Income must be greater than 0")
