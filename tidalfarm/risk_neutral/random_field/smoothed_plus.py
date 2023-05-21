import numpy as np

def smoothed_plus(x, mu):
    if x >= 0.0:
        return x+mu*np.logaddexp(0.0, -x/mu)
    else:
        return mu*np.logaddexp(x/mu, 0.0)
