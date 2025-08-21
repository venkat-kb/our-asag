import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error

def calculate_qwk(y_true, y_pred, max_rating=3):
    """
    Calculates the Quadratic Weighted Kappa (QWK) score.
    A metric that measures the agreement between two raters,
    penalizing larger differences in ratings more heavily.
    """
    # Ensure predictions are within the valid rating range [0, max_rating]
    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, 0, max_rating)
    
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def calculate_mae(y_true, y_pred):
    """Calculates the Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))