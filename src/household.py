from mesa import Agent
from mesa import Model
from typing import Tuple

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


def move_out_medium():
    p = 4 * (income_percentile - 0.5)**2
    assert 0 <= p <= 1
    return p