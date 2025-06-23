import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("data/agent_data.csv")

# Pivot: now Steps are rows, AgentIDs are columns
pivot = data.pivot(index="Step", columns="AgentID", values="income")

# Plot setup
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.08
steps = pivot.index.tolist()
x = list(range(len(steps)))  # One position per Step

# Agent IDs in order
agent_ids = pivot.columns.tolist()
n_agents = len(agent_ids)

# For each agent, plot their income across all timesteps
for i, agent in enumerate(agent_ids):
    offset = (i - n_agents / 2) * bar_width + bar_width / 2
    ax.bar(
        [xi + offset for xi in x],
        pivot[agent],
        width=bar_width,
        label=f"Agent {agent}"
    )

# Fix x-ticks to show Step numbers
ax.set_xticks(x)
ax.set_xticklabels([f"Step {s}" for s in steps])
ax.set_xlabel("Step")
ax.set_ylabel("Income")
ax.set_title("Agent Income per Timestep")
ax.legend(title="AgentID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fig/income_data_plot.png")
