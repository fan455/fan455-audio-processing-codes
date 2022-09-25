"""
A simple python code to normalize the peak volume of the mid channel of a stereo wav file under the normal -3db pan law.
"""
import numpy as np
import soundfile as sf

def amp2db(amp: float): # zero or positive amp value range between 0 and 1.
    return 20*np.log10(amp)

def db2amp(db: float): # zero or negative db value.
    return np.power(10, db/20)

def normalize_mid_path(input_path, output_path, amp=0.5, sf_subtype='PCM_24'):
    """
    Parameters:
    input_path: str. The path of input wav file.
    output_path: str. The path of output wav file (normalized).
    amp: float. The maximum amplitude of the mid channel of the output wav file, needs to be between 0 and 1. By default, amp is set to 0.5 (about -6db).
    sf_subtype: str. The subtype parameter for soundfile.write() function. By default it's 'PCM_24' which means 24-bit integer.
    
    If you want to normalize in db, use function db2amp to convert db to amplitude.
    This function can be easily used for batch processing.
    This function has no return but write to output_path instead. You can specify return if you need.
    """
    if not 0<=amp<=1:
        raise ValueError('0<=amp<=1 is needed.')
    au, sr = sf.read(input_path)
    scale = 2*amp/np.amax(np.abs(np.sum(au, axis=-1)))
    au *= scale
    sf.write(output_path, au, sr, subtype=sf_subtype)

def normalize_mid(au, amp=0.5):
    """
    input array, output array, no read or write audio files.
    """
    scale = 2*amp/np.amax(np.abs(np.sum(au, axis=-1)))
    au *= scale
    return au
