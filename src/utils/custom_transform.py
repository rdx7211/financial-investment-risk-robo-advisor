import numpy as np

def safe_log_transform(x):
    """Safe log transform: log1p(max(x, 0)) to avoid negative/zero issues."""
    return np.log1p(np.maximum(x, 0))
