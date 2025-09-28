import numpy as np

def safediv(a, b):
    """Safely divide numpy arrays a and b, taking into account zeroes."""
    return np.divide(a, b, out = np.zeros_like(b), where = b != 0)

def qadd(a, b):
    """Add two numbers / arrays in quadrature."""
    return np.sqrt(np.square(a) + np.square(b))