from mesa import Model


class Neighbourhood:
    def __init__(self, model: Model, x: int, y: int, rent_factor: float) -> None:
        """
        Initialize a neighbourhood with its coordinates, number of residents, total income, and rent factor.

        Parameters:
        - model (Model): The model instance to which this neighbourhood belongs.
        - x (int): X-coordinate of the neighbourhood.
        - y (int): Y-coordinate of the neighbourhood.
        - rent_factor (float): Factor by which the rent is calculated based on the neighbourhood's income.
        """

        self.x = x  # X-coordinate of the neighbourhood
        self.y = y  # Y-coordinate of the neighbourhood
        self.residents = 0  # Number  of residents in the neighbourhood
        self.total_income = 0  # Total income of the neighbourhood
        self.rent_factor = rent_factor  # Rent factor for the neighbourhood
        self.model = model

    def rent(self) -> float:
        """
        Calculate the rent for the neighbourhood based on the total income and number of residents.

        Returns:
        - float: The rent for the neighbourhood, capped at 100,000.
        """
        if len(self.model.neighbourhood_history) > 0:
            return min(
                self.model.neighbourhood_history[-1][self.x, self.y] * self.rent_factor,
                100_000,
            )

        if self.residents == 0:
            return 0
        return min((self.total_income / self.residents) * self.rent_factor, 100_000)
