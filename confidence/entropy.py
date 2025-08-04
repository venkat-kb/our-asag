import numpy as np
from scipy.stats import entropy

def compute_confidence(scores):
    scores = np.array(scores, dtype=np.float64)
    total = np.sum(scores)

    if total <= 0:
        probs = np.ones_like(scores) / len(scores)
    else:
        probs = scores / total

    conf = 1 - entropy(probs)

    # Clamp to [0, 1]
    return max(0.0, min(conf, 1.0))
