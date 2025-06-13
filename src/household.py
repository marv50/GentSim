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