from mesa import Agent
from mesa import Model
from typing import Tuple
import numpy as np


class Household(Agent):
    """
    A household agent in the GentSim model.
    """

    def __init__(self, model: Model, pos: tuple) -> None:
        super().__init__(model)
        self.income = 1  # Pooruhh
        self.income_bin = get_income_bin(self.income)
        self.pos = pos
        # self.ph

    def step(self, model: Model) -> None:
        """
        We movin or not
        """
        # I cant afford? -> MOVE
        if self.income_bin == "low":
            ...  # :( Pooruhh
        if self.income_bin == "medium":
            move_out_prob = move_out_medium(model, self.income, *self.pos)
            if model.random.random() < move_out_prob:
                ...
        if self.income_bin == "high":
            if model.random.random() < self.ph:
                self.move()

    def move(self) -> None: ...


def income_percentile(model, income, x, y) -> float:
    """
    Calculate the income percentile of the household.
    """

    assert income > 0, "Income must be greater than 0"
    neighbours = model.grid.get_neighbors((x, y), True, False)
    total = sum([n.income for n in neighbours])
    return income / (total + income)


def move_out_low(model, income, x, y) -> float:
    """
    Calculate the probability of moving out based on the income percentile.
    """
    gamma = income_percentile(model, income, x, y)
    p = 1 - np.sqrt(gamma)
    assert 0 <= p <= 1
    return p


def move_out_medium(model, income, x, y):
    """
    Calculate the probability of moving out based on the income percentile.

    ## Parameters
    - model (Model): The model instance.
    - income (float): The income of the household.
    - x (int): The x-coordinate of the household.
    - y (int): The y-coordinate of the household.

    ## Returns
    - float: The probability of moving out.
    """
    p = 4 * (income_percentile(model, income, x, y) - 0.5) ** 2
    assert 0 <= p <= 1
    return p


def move_in_medium(model, income, x, y) -> float:
    """
    Calculate the probability of moving in based on the income percentile.

    ## Parameters
    - model (Model): The model instance.
    - income (float): The income of the household.
    - x (int): The x-coordinate of the household.
    - y (int): The y-coordinate of the household.

    ## Returns
    - float: The probability of moving in.
    """
    p = 1 - move_out_medium(model, income, x, y)
    assert 0 <= p <= 1
    return p


def move_in_high(model, x, y) -> float:
    """
    Compute average income growth rate phi^epsilon(t) for a cell, required for h
    to move in somewhere else.
    """
    history = model.income_history[(x, y)]
    if len(history) < model.epsilon + 1:
        return 0.0
    diffs = [history[-(i + 1)] - history[-(i + 2)] for i in range(model.epsilon)]
    return sum(diffs) / model.epsilon


def move_in(model, utility_func, *args, **kwargs):
    """
    Calculate the utility of moving into a house.
    The utility is calculated as the inverse of the utility function.
    Where the utility function depends on the income level

    ## Parameters
    - model (Model): The model instance.
    - utility_func (function): The utility function to use for calculating the utility.
    - args: Additional arguments to pass to the utility function.
    - kwargs: Additional keyword arguments to pass to the utility function.

    ## Returns
    - float: The maximum rho value for the empty houses in the model.
    """

    empty_indices = np.argwhere(model.empty_houses)
    house_utilities = {
        (x, y): utility_func(model, *args, x=x, y=y, **kwargs)
        for (x, y) in empty_indices
    }
    total_sum = sum(house_utilities.values())
    houses_probs = {
        (x, y): value / (total_sum) if (total_sum - value) != 0 else 0
        for (x, y), value in house_utilities.items()
    }

    items = list(houses_probs.items())
    np.random.shuffle(items)  # Randomize order
    for (x, y), prob in items:
        if prob >= np.random.rand():
            model.empty_houses.remove((x, y))
            model.grid.place_agent(Household(model, (x, y)), (x, y))
            return 1
    return 0


def get_income_bin(income: float) -> int:
    """
    Get the income bin for the given income.

    ## Parameters
    - income (float): The income of the household.

    ## Returns
    - int: The income bin.
    """
    if income < 38690:
        return "low"
    elif 38690 < income < 77280:
        return "medium"
    elif income > 77280:
        return "high"
    else:
        raise ValueError("Income must be greater than 0")
