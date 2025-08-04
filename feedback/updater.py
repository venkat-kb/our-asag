import json
from collections import defaultdict
import numpy as np
from feedback.logger import log_update

LOG_PATH = "feedback/log.json"
PROMPT_CONFIG = "feedback/agent_weights.json"


def load_logs(path=LOG_PATH):
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def compute_agent_performance(logs):
    agent_scores = defaultdict(list)
    for entry in logs:
        for agent in entry["agents"]:
            agent_scores[agent["agent"].strip()] += [agent["score"]]
    return {agent: float(np.mean(scores)) for agent, scores in agent_scores.items()}

def write_config(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def rewrite_main(weights, agent_perf):
    import hashlib
    from datetime import datetime

    # Save weights to config file for use in main.py
    with open("feedback/agent_weights.json", "w") as f:
        json.dump({"weights": weights}, f, indent=2)

    q_hash = hashlib.sha256(json.dumps(agent_perf).encode()).hexdigest()[:8]
    a_hash = hashlib.sha256(str(weights).encode()).hexdigest()[:8]

    import hashlib
    from datetime import datetime
    with open("main.py", "r") as f:
        script_content = f.read()

    q_hash = hashlib.sha256(script_content.encode()).hexdigest()[:8]
    a_hash = hashlib.sha256(str(weights).encode()).hexdigest()[:8]

    log_update({
    "timestamp": datetime.utcnow().isoformat(),
    "weights": weights,
    "question_hash": q_hash,
    "answer_hash": a_hash,
    "agent_performance": agent_perf
    })

if __name__ == "__main__":
    logs = load_logs()
    agent_perf = compute_agent_performance(logs)
    write_config(agent_perf, PROMPT_CONFIG)

    # Apply weights to main.py
    agent_list = ["strict", "semantic"]
    weights = [agent_perf.get(a, 1.0) for a in agent_list]
    rewrite_main(weights, agent_perf)