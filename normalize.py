"""
A simple python code to normalize the peak volume of the mid channel of a stereo wav file under the normal -3db pan law.
"""
import numpy as np

def amp2db(amp: float): # zero or positive amp value range between 0 and 1.
    return 20*np.log10(amp)

def db2amp(db: float): # zero or negative db value.
    return np.power(10, db/20)

def normalize_mid(au, amp=0.35):
    """
    input array, output array, no read or write audio files.
    """
    scale = 2*amp/np.amax(np.abs(np.sum(au, axis=-1)))
    au *= scale
    return au

def normalize_mono(au, amp=0.35):
    """
    input array, output array, no read or write audio files.
    """
    scale = amp/np.amax(np.abs(au))
    au *= scale
    return au
