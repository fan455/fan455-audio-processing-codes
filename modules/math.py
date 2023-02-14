import numpy as np
from scipy.linalg import lstsq
from scipy.interpolate import CubicSpline, PchipInterpolator

def itp(x, px, py, axis=-1):
    return CubicSpline(px, py, axis=axis)(x)

def itpmono(x, px, py, axis=-1):
    return PchipInterpolator(px, py, axis=axis)(x)

def ols1v(px, py):
    M = np.vstack((np.ones(px.size), px)).T
    p, res, rnk, s = lstsq(M, py)
    return p[0], p[1]
