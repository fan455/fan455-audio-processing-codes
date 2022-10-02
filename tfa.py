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
        raise ValueError('Parameter "output" has to be "m, p", "z" or "r, i".')

def cent2ratio(cent):
    return np.exp2(cent/1200)

def get_pitch_given(au, sr, channel=0, du=None, given_freq=440, given_cent=50, f_step=1):
    if au.ndim == 1:
        au = au.reshape((au.size, 1))
    elif au.ndim == 2:
        au = au[:, channel:channel+1]
    else:
        raise ValueError('The input audio array has no dimension, or over 2 dimensions which means it may be a framed audio.')
    if du == None:
        t_size = sr*(au.size//sr)
    else:
        t_size = int(sr*du)
    au = au[0: t_size]
    t = (np.arange(0, t_size)/sr).reshape((t_size, 1))
    given_ratio = cent2ratio(given_cent)
    f_l, f_h = int(given_freq/given_ratio), int(given_freq*given_ratio) + 1
    f = np.arange(f_l, f_h+1, f_step)
    f_size = f.size
    t = np.broadcast_to(t, (t_size, f_size))
    m = np.abs(np.average(au*np.exp(-2*np.pi*f*1.0j*t), axis=0))
    pitch = f[np.argmax(m)]
    print(f'{round(pitch, 2)}Hz is the detected pitch given {round(given_freq, 2)}Hz, [{f_l}Hz, {f_h}Hz] band and {np.round(f_step, 2)}Hz step.')
    return pitch
