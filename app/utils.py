# app/utils.py
# Contains helper functions

import numpy as np

def safe_prediction(pred, fallback):
    """Replace NaN predictions with a fallback value."""
    return np.where(np.isnan(pred), fallback, pred)
