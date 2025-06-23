import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rv_discrete

plt.style.use("bmh")


def load_distribution(
    file_name: str = "data/income_data.csv", header=None
) -> pd.DataFrame:
    """
    Load the income distribution as a discrete random variable.
    """
    assert file_name.endswith(".csv"), "File must be a CSV file."
    df = pd.read_csv(file_name, header=header)
    df.columns = ["income_range", "frequency"]
    df["income_point"] = df["income_range"].apply(extract_points)
    df["probability"] = df["frequency"] / df["frequency"].sum()
    return df


def extract_points(range_str):
    """
    Extract the income point from a range string like "0-1000" or "1000-2000".
    Returns the lower bound of the range as an integer.
    """
    match = re.findall(r"\d+", range_str)
    return int(match[0])


def repeat_data(df: pd.DataFrame) -> pd.Series:
    """
    Repeat the income points according to their frequency.
    """
    expanded_data = np.repeat(df["income_point"], df["frequency"])
    return pd.Series(expanded_data)


def create_income_distribution(df: pd.DataFrame) -> rv_discrete:
    income_distribution = rv_discrete(
        name="income_dist", values=(df["income_point"], df["probability"])
    )
    return income_distribution
