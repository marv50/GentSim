import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rv_discrete


def load_distribution(
    file_name: str = "data/income_data.csv", header=None
) -> pd.DataFrame:
    """
    Load the income distribution as a discrete random variable.

    Parameters:
    - file_name (str): Path to the CSV file containing income ranges and frequencies.
    - header (int or None): Row number to use as the column names, or None for no header.

    Returns:
    - pd.DataFrame: DataFrame with income ranges, frequencies, income points, and probabilities
    """
    assert file_name.endswith(".csv"), "File must be a CSV file."
    df = pd.read_csv(file_name, header=header)
    df.columns = ["income_range", "frequency"]
    df["income_point"] = df["income_range"].apply(extract_points)
    df["probability"] = df["frequency"] / df["frequency"].sum()
    return df


def extract_points(range_str: str) -> int:
    """
    Extract the income point from a range string like "0-1000" or "1000-2000".
    Returns the lower bound of the range as an integer.

    Parameters:
    - range_str (str): The income range string.

    Returns:
    - int: The lower bound of the income range as an integer.
    """
    match = re.findall(r"\d+", range_str)
    return int(match[0])


def repeat_data(df: pd.DataFrame) -> pd.Series:
    """
    Repeat the income points according to their frequency.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'income_point' and 'frequency

    Returns:
    - pd.Series: Series with income points repeated according to their frequency.
    """
    expanded_data = np.repeat(df["income_point"], df["frequency"])
    return pd.Series(expanded_data)


def create_income_distribution(df: pd.DataFrame) -> rv_discrete:
    """
    Create a discrete random variable representing the income distribution.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'income_point' and 'probability'.

    Returns:
    - rv_discrete: A discrete random variable representing the income distribution.
    """
    income_distribution = rv_discrete(
        name="income_dist", values=(df["income_point"], df["probability"])
    )
    return income_distribution


def custom_income_distribution(
    N_agents: int, bin_probabilities: list, edges: list
) -> np.ndarray:
    """
    Create a custom income distribution for the agents.

    Parameters:
    - N_agents (int): Total number of agents.
    - bin_probabilities (list): List of probabilities for each income bin.
    - edges (list): List of edges defining the income bins.

    Returns:
    - np.ndarray: An array representing the income distribution of the agents.
    """
    assert (
        len(bin_probabilities) == 3
    )  # Ensure there are three probabilities for the bins
    assert len(bin_probabilities) == len(edges) - 1

    counts = np.random.multinomial(N_agents, bin_probabilities)
    income_distribution = np.hstack(
        [np.random.randint(edges[i], edges[i + 1], n) for i, n in enumerate(counts)]
    )
    return income_distribution
