import numpy as np

def separate_stereo_channels(au):
    return au[:, 0], au[:, 1]

