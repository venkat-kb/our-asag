import json
import os
import hashlib
from collections import defaultdict
import numpy as np
from datetime import datetime
from feedback.logger import log_update

LOG_PATH = "feedback/log.json"
PROMPT_CONFIG = "feedback/agent_weights.json"


def load_logs(path=LOG_PATH):
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def compute_agent_performance(logs, smoothing_factor=None):
    if smoothing_factor is None:
        smoothing_factor = float(os.environ.get("EMA_SMOOTHING", 0.5))

    max_weight = float(os.environ.get("DYNAMIC_CAP", 10.0))

    agent_errors = defaultdict(list)
    for entry in logs:
        true_score = entry.get("true_score")
        if true_score is None:
            continue
        for agent in entry["agents"]:
            agent_name = agent["agent"].strip()
            error = abs(agent["score"] - true_score)
            agent_errors[agent_name].append(error)

    try:
        with open(PROMPT_CONFIG, "r") as f:
            loaded = json.load(f)
            prev_weights = {k: v for k, v in zip(["strict", "semantic"], loaded.get("weights", []))}
    except:
        prev_weights = {}

    agents = list(agent_errors.keys())

    # Add cap here
    raw_weights = {
        agent: min(1.0 / (np.mean(agent_errors[agent]) + 1e-6), max_weight)
        for agent in agents
    }

    smoothed = {}
    for i, agent in enumerate(agents):
        prev = prev_weights.get(agent, raw_weights[agent])
        smoothed[agent] = smoothing_factor * raw_weights[agent] + (1 - smoothing_factor) * prev
        smoothed[agent] = max(0.0, smoothed[agent])

    return smoothed


def rewrite_main(weights, agent_perf):
    if not os.path.exists("feedback/agent_weights.json") or os.path.getsize("feedback/agent_weights.json") == 0:
        with open("feedback/agent_weights.json", "w") as f:
            json.dump({"weights": [1.0, 1.0]}, f, indent=2)
    with open("feedback/agent_weights.json", "w") as f:
        ordered = weights
        json.dump({"weights": ordered}, f, indent=2)

    q_hash = hashlib.sha256(json.dumps(agent_perf).encode()).hexdigest()[:8]
    a_hash = hashlib.sha256(str(weights).encode()).hexdigest()[:8]

    log_update({
        "timestamp": datetime.utcnow().isoformat(),
        "weights": weights,
        "question_hash": q_hash,
        "answer_hash": a_hash,
        "agent_performance": agent_perf
    })
