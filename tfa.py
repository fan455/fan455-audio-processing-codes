"""
Audio Time-Frequency Analysis codes based on numpy, scipy and matplotlib.
"""
import numpy as np
from scipy import signal

def psd(au, sr, channel=0, nperseg=None, noverlap=None):
    if au.ndim == 1:
        pass
    elif au.ndim == 2:
        au = au[:, channel]
    else:
        raise ValueError('The input audio array has no dimension, or over 2 dimensions which means it may be a framed audio.')
    if nperseg == None:
        nperseg = sr
    f, Pxx = signal.welch(au, fs=sr, nperseg=nperseg, noverlap=noverlap)
    return f, Pxx

def stft(au, sr, channel=0, output='m', nperseg=None, noverlap=0):
    """
    Parameters:
    au: numpy.ndarray. Need to have 1 or 2 dimensions like normal single-channel or multi-channel audio. win_idxd audio needs to be converted to non-win_idxd audio first using other functions to have the right stft.
    sr: int. Sample rate of au.
    channel: int. If au has 2 dimensions, which channel to do stft. Defaults to 0 which represents the first channel. 
    output: str. 'm' will return magnitudes array (zero or positive real values). 'm, p' will return two magnitudes and phases arrays. 'z' will return a complex array as scipy. 'r, i' will return two real and imaginary arrays derived from the complex array.
    nperseg: None or int. As scipy.signal.stft. None will use the sample rate, int is as scipy.signal.stft.
    noverlap: None or int. As scipy.signal.stft. None will use half of nperseg.
    
    Returns:
    f: as scipy.signal.stft returns.
    t: as scipy.signal.stft returns.
    m: if output='m'.
    m, p: if output='m, p'.
    z: if output='z'.
    z.real, z.imag: if output='r, i'.
    """
    if au.ndim == 1:
        pass
    elif au.ndim == 2:
        au = au[:, channel]
    else:
        raise ValueError('The input audio array has no dimension, or over 2 dimensions which means it may be a framed audio.')
    if nperseg == None:
        nperseg = sr
    if noverlap == None:
        noverlap = nperseg/2
    f, t, z = signal.stft(au, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    t = np.around(t, 2)
    if output == 'm':
        m = np.abs(z)
        return f, t, m
    elif output == 'm, p':
        m = np.abs(z)
        p = np.angle(z*np.exp((np.pi/2)*1.0j))
        return f, t, m, p
    elif output == 'z':
        return f, t, z
    elif output == 'r, i':
        return f, t, z.real, z.imag
    else:
        raise ValueError('Parameter "output" has to be "m, p", "complex" or "r, i".')

def cent2ratio(cent):
    return np.exp2(cent/1200)

def get_pitch(au, sr, channel=0, win_idx=0, given=True, given_freq=None, given_cent=None):
    """
    Search around the given frequency within the freq to get the more exact pitch. For piano single note sound.

    Parameters:
    channel: int. If audio is multi-channel, which channel to analyze.
    win_idx: int. If more than 1 stft window, which window to analyze.
    given: bool. Search with given frequency or not. This may be needed because pianos' lower notes may have some significant inharmonic frequencies caused by strings' logitudinal vibration.
    given_freq: float (Hz). One possible pitch given, usually the standard pitch. 
    given_cent: float (cent). Half of the cent band to search around the given pitch for the index of the frequency with maxinum amplitude.
    When given is True, both given_freq and given_cent cannot be None.
    """
    if au.ndim == 1:
        pass
    elif au.ndim == 2:
        au = au[:, channel]
    else:
        raise ValueError('au.ndim needs to be 1 or 2.')
    f, t, m = stft(au, sr)
    m = m[:, win_idx].reshape(f.size)
    if given:
        if given_freq != None and given_cent != None:
            given_ratio = cent2ratio(given_cent)
            f_l, f_h = given_freq/given_ratio, given_freq*given_ratio
            f_l_idx, f_h_idx = np.argmin(np.abs(f-f_l)), np.argmin(np.abs(f-f_h))
            f_cut, m_cut = f[f_l_idx: f_h_idx+1], m[f_l_idx: f_h_idx+1]
            pitch = f_cut[np.argmax(m_cut)]
            return pitch
        else:
            raise ValueError('When given is True, both given_freq and given_cent cannot be None.')
    else:
        pitch = f[np.argmax(m)]
        return pitch
