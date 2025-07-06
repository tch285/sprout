import numpy as np

def quad_log(x, x0, y0, a):
    return a * ((np.log(x/x0)) ** 2) + y0
def gaus_log(x, mu, C, sg):
    return C * np.exp(-(np.log(x/mu)) ** 2 / (2 * sg * sg))
def cust_TC(x, T, C, exp):
    return C * x / (x ** 2 + T) ** exp
def cust_tp(x, t, p, exp):
    return 3 * p * t**2 * np.sqrt(3) * x / (x ** 2 + 2 * t**2) ** exp

def get_ff(name):
    if name in ['quad', 'quad_log']:
        return quad_log
    elif name in ['gaus', 'gaus_log']:
        return gaus_log
    elif name in ['cust_TC']:
        return cust_TC
    elif name in ['cust_tp']:
        return cust_tp
    else:
        raise ValueError(f"Function name '{name}' not recognized.")