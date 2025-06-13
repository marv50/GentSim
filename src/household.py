from mesa import Agent
from mesa import Model
from typing import Tuple
import numpy as np

class Household(Agent):
    """
    A household agent in the GentSim model.
    """
    def __init__(self, model: Model) -> None:
        super().__init__(model)
        self.income = 0  # Pooruhh

    def step(self, model: Model) -> None:
        """ 
        We movin or not
        """
        #I cant afford? -> MOVE
        pass

def income_percentile(model, income, x, y) -> float:
    """
    Calculate the income percentile of the household.
    """
    assert income > 0,"Income must be greater than 0"
    neighbours = model.grid.get_neighbours((x, y), True, False)
    total = sum([n.income for n in neighbours])
    return income / (total + income)


def move_out_medium():
    p = 4 * (income_percentile() - 0.5)**2
    assert 0 <= p <= 1
    return p

def move_in_medium():
    houses = {
        (x, y)
    }