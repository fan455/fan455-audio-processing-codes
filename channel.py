"""
Separate or combine audio channels
"""
import numpy as np

def sep2to2(au):
    return au[:, 0], au[:, 1]

def sep4to4(au):
    return au[:, 0], au[:, 1], au[:, 2], au[:, 3]

def sep4to2(au):
    return au[:, 0:2], au[:, 2:4]

def comb2(au_L, au_R):
    return np.stack((au_L, au_R), axis=-1)
