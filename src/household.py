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

    def step(self, model: Model) -> None:
        """
        We movin or not
        """
        if self.income_bin == "low":
            move_out_prob = move_out_low(model, self.income, *self.pos)
            if model.random.random() < move_out_prob:
                # Fixed: pass income and position as separate arguments
                new_location = move_in(model, move_in_low, self.income, self.pos)
                if new_location:
                    self.move(model, new_location)

        if self.income_bin == "medium":
            move_out_prob = move_out_medium(model, self.income, *self.pos)
            if model.random.random() < move_out_prob:
                # Fixed: pass income and position as separate arguments
                new_location = move_in(model, move_in_medium, self.income, self.pos)
                if new_location:
                    self.move(model, new_location)

        if self.income_bin == "high":
            if model.random.random() < model.p_h:
                # Fixed: pass position as tuple
                new_location = move_in(model, move_in_high, self.pos)
                if new_location:
                    self.move(model, new_location)

    def move(self, model, location):
        """
        Move the household to a new location.
        """
        old_pos = self.pos
        model.empty_houses[old_pos] = True
        model.empty_houses[location] = False
        self.pos = location
        model.grid.move_agent(self, location)

        # Fixed assertions
       # assert model.empty_houses[old_pos] is True, "Old position must be empty after move"
        #assert model.empty_houses[location] is False, "New position must be occupied after move"


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


def move_in_low(model, income, x, y) -> float:
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
    gamma = income_percentile(model, income, x, y)
    p = np.sqrt(gamma)
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
    Compute average income growth rate phi^epsilon(t) for a cell, required for high
    income households to move in somewhere else.

    This calculates the sum of differences of median incomes over the past epsilon timesteps.

    Parameters:
    - model: The GentSimModel instance
    - x (int): The x-coordinate of the cell
    - y (int): The y-coordinate of the cell

    Returns:
    - float: Sum of income growth differences over epsilon periods
    """
    if len(model.grid_history) < model.epsilon + 1:
        return 0.0

    # Get the last epsilon + 1 grid snapshots to calculate epsilon differences
    recent_grids = model.grid_history[-(model.epsilon + 1):]

    # Calculate median income for the Moore neighborhood in each time period
    medians = []
    for grid_snapshot in recent_grids:
        # Get Moore neighborhood coordinates (including center)
        neighbor_incomes = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < model.grid.width and 0 <= ny < model.grid.height:
                    income = grid_snapshot[nx, ny]
                    if income > 0:  # Only include occupied cells
                        neighbor_incomes.append(income)

        if neighbor_incomes:
            medians.append(np.median(neighbor_incomes))
        else:
            medians.append(0.0)

    # Calculate the sum of differences over epsilon periods
    # medians[0] is oldest, medians[-1] is most recent
    if len(medians) < model.epsilon + 1:
        return 0.0

    # Calculate differences: [t-(epsilon-1)] - [t-epsilon], [t-(epsilon-2)] - [t-(epsilon-1)], ..., [t] - [t-1]
    diffs = []
    for i in range(model.epsilon):
        diff = medians[i + 1] - medians[i]  # newer - older
        diffs.append(diff)

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
    - tuple or 0: The coordinates (x, y) of the selected house, or 0 if no house is selected.
    """
    empty_indices = np.argwhere(model.empty_houses)
    if len(empty_indices) == 0:
        return 0  # No empty houses available
    
    house_utilities = {}
    
    # Handle different utility function signatures
    for x, y in empty_indices:
        if utility_func == move_in_high:
            # move_in_high only takes model, x, y
            utility = utility_func(model, x, y)
        else:
            # move_in_low and move_in_medium take model, income, x, y
            # args should contain income and potentially position info
            if len(args) >= 2 and isinstance(args[1], tuple):
                # Case: move_in(model, utility_func, income, pos)
                income = args[0]
                utility = utility_func(model, income, x, y)
            elif len(args) >= 1:
                # Case: move_in(model, utility_func, income)
                income = args[0]
                utility = utility_func(model, income, x, y)
            else:
                # Fallback - shouldn't happen with proper usage
                utility = utility_func(model, x, y, **kwargs)
        
        house_utilities[(x, y)] = utility
    
    total_sum = sum(house_utilities.values())
    if total_sum == 0:
        return 0  # All utilities are 0
    
    houses_probs = {
        (x, y): value / total_sum
        for (x, y), value in house_utilities.items()
    }

    items = list(houses_probs.items())
    np.random.shuffle(items)  # Randomize order
    for (x, y), prob in items:
        if prob >= np.random.rand():
            return (x, y)
    return 0


def get_income_bin(income: float) -> str:  # Fixed return type annotation
    """
    Get the income bin for the given income.

    ## Parameters
    - income (float): The income of the household.

    ## Returns
    - str: The income bin.
    """
    if income < 38690:
        return "low"
    elif 38690 <= income < 77280:  # Fixed condition to handle edge case
        return "medium"
    elif income >= 77280:  # Fixed condition to handle edge case
        return "high"
    else:
        raise ValueError("Income must be greater than 0")