import numpy as np
from scipy.stats import entropy

def compute_confidence(scores):
    probs = np.array(scores) / np.sum(scores)
    conf = 1 - entropy(probs)
    return conf
