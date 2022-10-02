"""
Normalize mono or stereo audio.
"""
import numpy as np

def normalize_mid(au, amp=0.35):
    """
    Normalize the peak volume of the mid channel of a stereo wav file under the normal -3db pan law.
    input array, output array, no read or write audio files.
    """
    scale = 2*amp/np.amax(np.abs(np.sum(au, axis=-1)))
    au *= scale
    return au

def normalize_mono(au, amp=0.35):
    """
    Normalize the peak volume of a mono audio.
    input array, output array, no read or write audio files.
    """
    scale = amp/np.amax(np.abs(au))
    au *= scale
    return au
