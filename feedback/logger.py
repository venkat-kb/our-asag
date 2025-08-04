import os
import json
from pathlib import Path

def log_feedback(data, file="feedback/log.json"):
    Path("feedback").mkdir(exist_ok=True)
    with open(file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def log_update(data, file="feedback/update_log.json"):
    Path("feedback").mkdir(exist_ok=True)
    with open(file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())