from mesa import Agent
from mesa import Model
from typing import Tuple
import numpy as np

class Household(Agent):
    """
    A household agent in the GentSim model.
    """
    def __init__(self, unique_id, model: Model, pos: tuple) -> None:
        super().__init__(unique_id, model)
        self.income = 1  # Pooruhh
        self.pos = pos

    def step(self, model: Model) -> None:
        """ 
        We movin or not
        """
        temp = move_in(model, move_in_medium, self.income)
        print(f"Move in probability: {temp}")


def income_percentile(model, income, x, y) -> float:
    """
    Calculate the income percentile of the household.
    """

    assert income > 0,"Income must be greater than 0"
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

def move_in_medium(model, income, x, y) -> float:
    """
    Calculate the probability of moving in based on the income percentile.
    """
    p = 1 - move_out_low(model, income, x, y)
    assert 0 <= p <= 1
    return p


def move_out_medium(model, income, x, y):
    """
    Calculate the probability of moving out based on the income percentile.
    """
    p = (4 * (income_percentile(model, income, x, y) - 0.5)**2)
    assert 0 <= p <= 1
    return p

def move_in_medium(model, income, x, y) -> float:
    """
    Calculate the probability of moving in based on the income percentile.
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
    diffs = [history[-(i + 1)] - history[-(i + 2)]
             for i in range(model.epsilon)]
    return sum(diffs) / model.epsilon


def move_in(model, utility_func, *args, **kwargs):
    """
    Calculate the utility of moving into a house.
    The utility is calculated as the inverse of the utility function.
    Where the utility function depends on the income level
    """

    empty_indices = np.argwhere(model.empty_houses)
    house_utilities = {
        (x, y): utility_func(model, *args, x=x, y=y, **kwargs) for (x,y) in empty_indices
    }
    total_sum = sum(house_utilities.values())
    rho = np.max([
        value / (total_sum - value) if (total_sum - value) != 0 else 0
        for value in house_utilities.values()
    ])

    # assert 0 <= rho <= 1
    return rho
