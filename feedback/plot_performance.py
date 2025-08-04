import json
import matplotlib.pyplot as plt
from datetime import datetime

def load_update_log(file="feedback/update_log.json"):
    data = []
    with open(file, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            if "agent_performance" in entry and "weights" in entry:
                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                data.append(entry)
    return data

def plot_agent_performance(data):
    agent_scores = {}
    weight_history = {}
    timestamps = [entry["timestamp"] for entry in data]

    for entry in data:
        for agent, score in entry["agent_performance"].items():
            if agent not in agent_scores:
                agent_scores[agent] = []
            agent_scores[agent].append(score)
        for i, weight in enumerate(entry["weights"]):
            key = f"weight_agent_{i}"
            if key not in weight_history:
                weight_history[key] = []
            weight_history[key].append(weight)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for agent, scores in agent_scores.items():
        ax1.plot(timestamps, scores, label=f"{agent} score", marker='o')
    ax1.set_ylabel("Mean Agent Score")
    ax1.legend()
    ax1.grid(True)

    for key, values in weight_history.items():
        ax2.plot(timestamps, values, label=f"{key}", linestyle='--', marker='x')
    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("Agent Weight")
    ax2.legend()
    ax2.grid(True)

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("feedback/performance_plot.png")
    print("📈 Saved graph to feedback/performance_plot.png")
