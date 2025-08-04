import json
from pathlib import Path

def log_feedback(data, file="feedback/log.json"):
    Path("feedback").mkdir(exist_ok=True)
    with open(file, "a") as f:
        f.write(json.dumps(data) + "\n")