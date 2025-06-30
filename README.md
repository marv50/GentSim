# GentSim
A Python library for running an Agent-Based Model (ABM) that simulates gentrification dynamics in urban neighborhoods.

**Authors:**
- Marvin Frommer (15905756)
- Luke Kraakman (13690868)
- Tycho Stam (13303147)
- Fabian Ivulic (14016273)

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Usage and Installation](#usage-and-installation)
4. [Implementation](#implementation)
5. [Scripts and Analysis](#scripts-and-analysis)
6. [Data and Outputs](#data-and-outputs)
7. [License](#license)

## Overview

GentSim is a sophisticated Agent-Based Model designed to simulate and analyze gentrification processes in urban environments. The model implements household agents with varying income levels that interact within a spatial grid representing neighborhoods and housing units. Through these interactions, the model captures complex socio-economic dynamics including:

- **Residential mobility patterns** based on income and affordability
- **Neighborhood change** through rent dynamics and income composition
- **Spatial segregation** and clustering effects
- **Income-based displacement** and gentrification processes

The simulation uses the Mesa framework for agent-based modeling and includes comprehensive tools for data analysis, visualization, and sensitivity analysis using Morris sampling methods.

## Features

- **Multi-income household agents** with realistic income distributions based on Dutch data
- **Dynamic rent calculation** based on neighborhood income composition and historical trends  
- **Spatial neighborhood structure** with configurable grid sizes and Moore neighborhoods
- **Income-based mobility decisions** with different utility functions for low, medium, and high-income households
- **Historical memory** for income trends affecting high-income household decisions
- **Comprehensive data collection** and CSV export capabilities
- **Advanced visualization tools** for grid evolution, income distribution, and clustering analysis
- **Parameter sweep functionality** with parallel execution
- **Sensitivity analysis** using SALib (Morris method)
- **Statistical analysis tools** for clustering, spatial disparity, and income dynamics

## Usage and Installation

To get started, first clone this repository:

```sh
gh repo clone https://github.com/marv50/GentSim.git
cd GentSim
```

This project uses [uv](https://docs.astral.sh/uv/) for Python package and environment management. Python 3.13+ is required. To ensure consistent results, we recommend using uv when running experiments. It is also possible to run experiments using [venv](https://docs.python.org/3/library/venv.html), though care must be taken to set up the Python environment correctly.

### Usage (uv) - Recommended

1. Configure Python environment and install any required dependencies:

    ```sh
    uv sync
    ```

2. Run basic simulation:

    ```sh
    uv run main.py
    ```

3. Run multiple simulations with parameter sets:

    ```sh
    uv run scripts/run_parameter_set.py
    ```

4. Perform parameter sweeps for sensitivity analysis:

    ```sh
    uv run scripts/run_sweep.py
    ```

### Usage (venv)

> [!IMPORTANT]
> These instructions assume that you have Python 3.13+ available 
> (see version specifier in [pyproject.toml](pyproject.toml)).
> In the following instructions we refer to the Python executable as `python`, 
> however, this may differ on your machine.

1. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Initialize directories as packages (if needed):

    ```sh
    pip install -e .
    ```

4. Run simulations:

    ```sh
    python main.py
    ```

## Implementation

The GentSim model consists of several key components organized in a modular structure:

### Source Code (`src/`)

#### Core Model (`model.py`)
- **GentSimModel**: Main model class implementing the Mesa framework
- Grid initialization with configurable neighborhoods and house counts
- Agent population initialization with income distributions
- Historical data tracking for income trends and neighborhood changes
- DataCollector integration for comprehensive data export

#### Household Agents (`household.py`)
- **Household**: Agent class representing individual households
- Income-based categorization (low, medium, high income)
- Sophisticated mobility decisions using utility functions:
  - **Low-income households**: Move out probability based on income percentile (√γ function)
  - **Medium-income households**: Quadratic utility function favoring moderate income diversity
  - **High-income households**: Historical income growth analysis with ε-memory
- Affordability constraints and neighborhood rent calculations

#### Neighborhoods (`neighbourhood.py`)
- **Neighbourhood**: Spatial units containing multiple houses
- Dynamic rent calculation based on average neighborhood income
- Resident and income tracking for each neighborhood
- Rent factor multiplication for realistic pricing

#### Income Distribution (`income_distribution.py`)
- Real Dutch income data processing from CSV files
- Custom income distribution generation with configurable bins
- Support for both empirical and synthetic income distributions
- Income categorization functions for agent classification

#### Data Processing (`csv_converter.py`)
- Conversion of simulation outputs to spatial time-series grids
- Multi-run data aggregation and standardization
- Position parsing and grid reconstruction
- 4D array generation (runs × time × height × width)

#### Data Analysis (`data_analysis.py`)
- **Clustering analysis**: Spatial segregation measurement using income bin concentrations
- **Income disparity**: Neighborhood-level inequality metrics
- **Temporal analysis**: Time-series tracking of key indicators
- **Parameter sweep analysis**: Batch processing of simulation results

#### Simulation Runner (`simulation_runner.py`)
- **Single simulations**: Individual model runs with parameter control
- **Multiple runs**: Parallel execution with result aggregation
- **Parameter sweeps**: SALib integration for Morris sensitivity analysis
- Comprehensive result storage and data management

### Scripts (`scripts/`)

#### Visualization (`create_plots.py`)
- **Grid evolution visualization**: Spatial income distribution over time
- **Income distribution plots**: Histogram generation from Dutch data
- **Clustering time-series**: Segregation dynamics visualization
- **Sensitivity analysis plots**: Morris method results (μ* and σ)
- **Spatial disparity trends**: Neighborhood inequality over time

#### Analysis Scripts
- **`run_parameter_set.py`**: Execute multiple runs with fixed parameters
- **`run_sweep.py`**: Parameter sensitivity analysis using Morris sampling
- **`sensitivity_analysis.py`**: SALib integration for comprehensive sensitivity analysis
- **`parallel_run.py`**: Batch execution using Mesa's BatchRunner
- **`plot_mu.py`**: Elementary effects visualization

## Scripts and Analysis

### Running Simulations

**Single Run:**
```sh
uv run main.py
```
Executes a single simulation with default parameters and generates basic visualizations.

**Multiple Runs:**
```sh
uv run scripts/run_parameter_set.py
```
Runs multiple simulations with identical parameters to analyze variability and generate statistical summaries.

**Parameter Sweep:**
```sh
uv run scripts/run_sweep.py
```
Performs sensitivity analysis using Morris sampling across parameter ranges:
- `epsilon`: Agent tolerance (2-10)
- `p_h`: Probability of moving (0.1-0.9)
- `b`: Bias parameter (0.0-1.0)
- `r_moore`: Neighborhood radius (1-2)
- `rent_factor`: Rent multiplier (0.3-0.7)

### Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `n_agents` | Number of household agents | 400 | 10-1000+ |
| `n_neighborhoods` | Grid neighborhoods per axis | 5 | 3-10 |
| `n_houses` | Houses per neighborhood side | 5 | 3-10 |
| `epsilon` | Historical memory length | 8 | 2-20 |
| `p_h` | High-income move probability | 0.4 | 0.1-0.9 |
| `b` | Local vs global preference | 0.5 | 0.0-1.0 |
| `r_moore` | Neighborhood interaction radius | 1 | 1-3 |
| `rent_factor` | Rent calculation multiplier | 0.7 | 0.3-1.0 |
| `steps` | Simulation timesteps | 50 | 10-200 |

### Analysis Methods

**Clustering Analysis:**
Measures spatial segregation by calculating income homogeneity within neighborhoods using the formula:
```
clustering = (n_low² + n_medium² + n_high²) / (n_low + n_medium + n_high)
```

**Spatial Income Disparity:**
Tracks inequality by measuring the difference between richest and poorest neighborhoods over time.

**Sensitivity Analysis:**
Uses Sobel method to identify parameter importance through first order and total order values.

## Data and Outputs

### Input Data
- **`data/income_data.csv`**: Real Dutch household income distribution (2022)
- Configuration parameters in scripts and main execution files

### Generated Outputs
- **`data/agent_data.csv`**: Single simulation agent-level data
- **`data/combined_agent_data.csv`**: Multi-run aggregated results
- **`data/sweep_results/`**: Parameter sweep outputs (one file per parameter set)

### Visualizations (`fig/`)
- **`income_distribution.png`**: Dutch income data histogram
- **`income_grid_evolution.png`**: Spatial household distribution over time
- **`clustering_over_time.png`**: Segregation dynamics
- **`morris_sensitivity_analysis.png`**: Parameter sensitivity results
- **`average_income_over_time.png`**: Temporal income trends

### Data Structure
Simulation outputs use a standardized format:
- **Agent data**: CSV with columns for AgentID, Step, income, pos, neighborhood
- **Grid data**: 3D/4D NumPy arrays (time × height × width) or (runs × time × height × width)
- **Analysis results**: Statistical summaries and visualization-ready datasets

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
