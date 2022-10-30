"""
Audio Time-Frequency Analysis codes based on numpy, scipy and matplotlib.
"""
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

def get_framed(au, sr, T=0.4, overlap=0.75, win='rectangular'):
    """
    Parameters
    au: ndarray. Needs to have shape mono (samples, ) or multi-channel (samples, channels)
    sr: float, Hz. Sample rate of input audio array.
    T: float, seconds. Time length of each window.
    overlap: float, proportion. Proportion of overlapping between windows.
    win: str. The window to apply to every frame.

    Returns
    au_f: ndarray. Framed audio with shape mono (window_num, samples) or multi-channel (window_num, samples, channels).
    """
    step = int(sr*T)
    hop = int(step*(1-overlap))
    if au.ndim == 2:
        q1, q2 = divmod(au.shape[0], hop)
        q3 = step - hop - q2
        if q3 > 0:
            au = np.append(au, np.zeros((q3, au.shape[-1])), axis=0)
        elif q3 < 0:
            raise ValueError('q3 < 0')
        au = au.reshape((1, au.shape[0], au.shape[1]))
        au_f = au[:, 0: step, :]
        for i in range(1, q1):
            au_f = np.append(au_f, au[:, i*hop: i*hop+step, :], axis=0)
        if win == 'rectangular':
            pass
        elif win == 'hanning':
            au_f *= np.broadcast_to(np.hanning(step).reshape((1, step, 1)), au_f.shape)
        elif win == 'hamming':
            au_f *= np.broadcast_to(np.hamming(step).reshape((1, step, 1)), au_f.shape)
        else:
            raise ValueError(f'window "{win}" is not supported.')
        return au_f
    elif au.ndim == 1:
        q1, q2 = divmod(au.shape[0], hop)
        q3 = step - hop - q2
        if q3 > 0:
            au = np.append(au, np.zeros(q3), axis=0)
        elif q3 < 0:
            raise ValueError('q3 < 0')
        au = au.reshape((1, au.shape[0]))
        au_f = au[:, 0: step]
        for i in range(1, q1):
            au_f = np.append(au_f, au[:, i*hop: i*hop+step], axis=0)
        if win == 'rectangular':
            pass
        elif win == 'hanning':
            au_f *= np.broadcast_to(np.hanning(step).reshape((1, step)), au_f.shape)
        elif win == 'hamming':
            au_f *= np.broadcast_to(np.hamming(step).reshape((1, step)), au_f.shape)
        else:
            raise ValueError(f'window "{win}" is not supported.')
        return au_f
    else:
        raise ValueError(f'au.ndim = {au.ndim} is not supported.')

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

def get_pitch_given(au, sr, channel=0, du=None, given_freq=440, given_cent=100, cent_step=1):
    """
    Detect the pitch of audio (specifically piano single note) given a pitch, cent band and cent step, using discrete time fourier transform in limited frequency range.
    The computation will be a bit slow since it does not use FFT, but it's way more accurate than scipy.signal.stft in terms of frequency resolution. 
    I've ensured the cpu and memory pressure won't be high using for-loop instead of 2d array.
    
    Parameters:
    au: ndarray (float between -1 and 1). The input audio.
    sr: int. Sample rate of audio.
    channel: int. The index of the audio channel to analyze. Only supports 1-channel analysis.
    du: None or float (seconds). The duration of audio to be analyzed. If set to None, it will be the maxinum integar seconds available.
    given_freq: float (Hz).
    given_cent: positive float (cent). Half of the cent band around the given frequency for pitch detection.
    cent_step: float (cent). The distance between Fourier transform's frequencies measured in cents.
    """
    if au.ndim == 1:
        pass
    elif au.ndim == 2:
        au = au[:, channel]
    else:
        raise ValueError('The input audio array has no dimension, or over 2 dimensions which means it may be a framed audio.')
    if du == None:
        t_size = sr*(au.size//sr)
    else:
        t_size = int(sr*du)
    au = au[0: t_size]
    t = np.arange(0, t_size)/sr
    F = given_freq*cent2ratio(np.arange(-given_cent, given_cent+1, cent_step))
    F_size = F.size
    M = np.empty(0)
    for i in range(0, F_size):
        f = F[i]
        m = np.abs(np.average(au*np.exp(-2*np.pi*f*1.0j*t)))
        M = np.append(M, m)
    pitch = F[np.argmax(M)]
    print(f'{round(pitch, 2)}Hz is the detected pitch given {round(given_freq, 2)}Hz, {round(given_cent, 2)} cent band and {np.round(cent_step, 2)} cent step.')
    return pitch

def interpolate_pitch(f, num):
    """
    Interpolate a frequency array.

    Parameters:
    f: ndarray (Hz). The input frequency array, strictly increasing.
    num: int. Number of frequencies to interpolate between every 2 adjacent input frequencies.
    """
    size = (num + 1)*f.size - num
    n, n_f = np.arange(0, size), np.arange(0, size, num+1)
    cs = CubicSpline(n_f, f)
    f_itp = cs(n)
    return f_itp

def interpolate_pitch_midi(f, Midi_f, Midi):
    """
    f: ndarray (Hz). The input frequency array, strictly increasing.
    Midi_f: ndarray. The midi array corresponding to the f array.
    Midi: ndarray. The midi array to apply interpolation.
    """
    cs = CubicSpline(Midi_f, f)
    return cs(Midi)
