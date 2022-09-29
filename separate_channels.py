import numpy as np

def separate_2_channels(au):
    return au[:, 0], au[:, 1]

def separate_4_channels(au):
    return au[:, 0], au[:, 1], au[:, 2], au[:, 3]
