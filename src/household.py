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

    def step(self, model: Model) -> None:
        """
        We movin or not
        """
        if self.income_bin == "low":
            move_out_prob = move_out_low(model, self.income, self.pos)
            if model.random.random() >= move_out_prob: return
            new_location = move_in(
                model,
                utility_func=move_in_low,
                income=self.income,
            )
            if not new_location: return
            self.move(model, new_location)

        if self.income_bin == "medium":
            move_out_prob = move_out_medium(model, self.income, self.pos)
            if model.random.random() >= move_out_prob: return
            new_location = move_in(
                model,
                utility_func=move_in_medium,
                income=self.income,
            )
            if not new_location: return
            self.move(model, new_location)

        if self.income_bin == "high":
            if model.random.random() >= model.p_h: return
            new_location = move_in(
                model,
                utility_func=move_in_high,
            )
            if not new_location:return
            self.move(model, new_location)

    def move(self, model, location):
        """
        Move the household to a new location.
        """
        old_pos = self.pos
        model.empty_houses[old_pos] = True
        model.empty_houses[location] = False

        assert bool(model.empty_houses[self.pos]) is True, "Old position must be empty"
        assert bool(model.empty_houses[location]) is False, (
            "New position must not be empty"
        )

        # Update the neighbourhoods
        old_neighbourhood = model.neighbourhoods[tuple(ti // model.N_neighbourhoods for ti in old_pos)]
        new_neighbourhood = model.neighbourhoods[tuple(ti // model.N_neighbourhoods for ti in location)]

        old_neighbourhood.residents -= 1
        new_neighbourhood.residents += 1
        new_neighbourhood.total_income += self.income
        old_neighbourhood.total_income -= self.income

        model.grid.move_agent(self, location)


def income_percentile(model, income, pos, b = 0) -> float:
    """
    Calculate the income percentile of the household.
    """
    assert income > 0, "Income must be greater than 0"

    local_neighbours = model.grid.get_neighbors(pos, True, False)
    local_total = sum([n.income for n in local_neighbours])
    local_ip = income / (local_total + income) 

    chunk_total = model.neighbourhoods[tuple(ti // model.N_neighbourhoods for ti in pos)].total_income
    chunk_ip = income / chunk_total
    return b * chunk_ip + (1 - b) * local_ip


def move_out_low(model, income, pos) -> float:
    """
    Calculate the probability of moving out based on the income percentile.
    """
    gamma = income_percentile(model, income, pos)
    p = 1 - np.sqrt(gamma)
    #assert 0 <= p <= 1
    return p


def move_out_medium(model, income, pos):
    """
    Calculate the probability of moving out based on the income percentile.
    """
    p = 4 * (income_percentile(model, income, pos) - 0.5) ** 2
    #assert 0 <= p <= 1
    return p


def move_in_low(model, income, pos) -> float:
    """
    Calculate the probability of moving in based on the income percentile.
    """
    gamma = income_percentile(model, income, pos)
    p = np.sqrt(gamma)
    #assert 0 <= p <= 1
    return p


def move_in_medium(model, income, pos) -> float:
    """
    Calculate the probability of moving in based on the income percentile.
    """
    p = 1 - move_out_medium(model, income, pos)
    #assert 0 <= p <= 1
    return p


def move_in_high(model, pos) -> float:
    """
    Compute average income growth rate phi^epsilon(t) for a cell, required for high
    income households to move in somewhere else.
    """
    if len(model.grid_history) < model.epsilon + 1:
        return 0.0

    recent_grids = model.grid_history[-(model.epsilon + 1) :]

    neighbor_positions = model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=2)

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
        return 0.0

    # Calculate differences: [t-(epsilon-1)] - [t-epsilon], [t-(epsilon-2)] - [t-(epsilon-1)], ..., [t] - [t-1]
    diffs = []
    for i in range(model.epsilon):
        diff = medians[i + 1] - medians[i]  # newer - older
        diffs.append(diff)

    return sum(diffs) / model.epsilon

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


def get_income_bin(income: float) -> str:  # Fixed return type annotation
    """
    Get the income bin for the given income.
    """
    if income < 38690:
        return "low"
    elif 38690 <= income < 77280:  # Fixed condition to handle edge case
        return "medium"
    elif income >= 77280:  # Fixed condition to handle edge case
        return "high"
    else:
        raise ValueError("Income must be greater than 0")
