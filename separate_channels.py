import numpy as np

def sep2to2(au):
    return au[:, 0], au[:, 1]

def sep4to4(au):
    return au[:, 0], au[:, 1], au[:, 2], au[:, 3]

def sep4to2(au):
    return au[:, 0:2], au[:, 2:4]

