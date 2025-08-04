import json
import matplotlib.pyplot as plt
from datetime import datetime
import os


def load_update_log(file="feedback/update_log.json"):
    data = []
    with open(file, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            if "agent_performance" in entry:
                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                data.append(entry)
    return data


def load_full_logs(file="feedback/log.json"):
    data = []
    if os.path.exists(file):
        with open(file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except:
                    continue
    return data


def plot_agent_performance(update_data):
    agent_scores = {}
    timestamps = [entry["timestamp"] for entry in update_data]

    for entry in update_data:
        for agent, score in entry["agent_performance"].items():
            if agent not in agent_scores:
                agent_scores[agent] = []
            agent_scores[agent].append(score)

    plt.figure(figsize=(12, 6))
    for agent in agent_scores:
        scores = agent_scores[agent]
        # Pad with None for early timestamps where agent wasn't present
        padded_scores = [None] * (len(timestamps) - len(scores)) + scores
        plt.plot(timestamps, padded_scores, label=agent, marker='o')

    plt.xlabel("Timestamp")
    plt.ylabel("Smoothed Agent Performance")
    plt.title("Agent Performance Over Time")
    plt.legend()
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("feedback/performance_plot.png")
    print(" Saved agent performance graph to feedback/performance_plot.png")


def plot_final_vs_true(log_data):
    if not log_data:
        return

    # Filter valid entries
    data = [
        (i, entry["final_score"], entry["true_score"])
        for i, entry in enumerate(log_data)
        if "final_score" in entry and "true_score" in entry
    ]

    if not data:
        print(" No valid entries for final vs true plot.")
        return

    timestamps, final_scores, true_scores = zip(*data)

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, final_scores, label="Final Score", marker="o")
    plt.plot(timestamps, true_scores, label="True Score", marker="x")
    plt.fill_between(timestamps, final_scores, true_scores, color='gray', alpha=0.2, label="Error")

    plt.xlabel("Run Index")
    plt.ylabel("Score (0–3)")
    plt.title("Final Score vs True Score Over Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("feedback/final_vs_true_plot.png")
    print(" Saved final vs true score graph to feedback/final_vs_true_plot.png")
