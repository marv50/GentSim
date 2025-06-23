import matplotlib.pyplot as plt
import pandas as pd

from src.income_distribution import repeat_data, load_distribution

plt.style.use("bmh")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 18
FIG_DPI = 300


def plot_income_distribution(
    title: str = "Income Distribution",
    xlabel: str = "Income",
    ylabel: str = "Frequency",
    bins: int = 50,
    save=True,
    save_path: str = "fig/income_distribution.png",
):
    """
    Plot the income distribution.
    """
    df = load_distribution()

    expanded_data = repeat_data(df)
    pd.Series(expanded_data).plot(
        kind="hist", bins=bins, edgecolor="black", density=True
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
