class Neighbourhood:
    def __init__(self, model, x, y, rent_factor=1.0):
        self.x = x  # X-coordinate of the neighbourhood
        self.y = y  # Y-coordinate of the neighbourhood
        self.residents = 0  # Number  of residents in the neighbourhood
        self.total_income = 0  # Total income of the neighbourhood
        self.rent_factor = rent_factor  # Rent factor for the neighbourhood
        self.model = model

    def rent(self):
        """
        Calculate the rent for the neighbourhood based on the total income and number of residents.
        """
        if len(self.model.neighbourhood_history) > 0:
            return min(self.model.neighbourhood_history[-1][self.x, self.y] * self.rent_factor, 100_000)
        
        if self.residents == 0:
            return 0
        return min((self.total_income / self.residents) * self.rent_factor, 100_000)
    