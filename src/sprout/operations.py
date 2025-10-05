import numpy as np

def safediv(a, b):
    """Safely divide numpy arrays a and b, taking into account zeroes."""
    return np.divide(a, b, out = np.zeros_like(b), where = b != 0)

def cdiv(h1, h2, rho):
    """Divide two H1Ds, with some level of correlation."""
    res = h1 / h2
    res.yerr = np.abs(res.contents) * np.sqrt(h1.yerr_rel**2 + h2.yerr_rel**2 - 2 * rho * h1.yerr_rel * h2.yerr_rel)
    return res

def qadd(a, b):
    """Add two numbers / arrays in quadrature."""
    return np.sqrt(np.square(a) + np.square(b))
