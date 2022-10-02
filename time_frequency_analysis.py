"""
Audio Time-Frequency Analysis codes based on numpy, scipy and matplotlib.
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

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
    output: str. 'm' will return magnitudes array (zero or positive real values). 'm, p' will return two magnitudes and phases arrays. 'complex' will return a complex array as normal. 'r, i' will return two real and imaginary arrays derived from the complex array.
    nperseg: None or int: None will use the sample rate, int is as scipy.signal.stft.
    noverlap: int. As scipy.signal.stft.
    
    Returns:
    f: as scipy.signal.stft returns.
    t: as scipy.signal.stft returns.
    m: if output='m'.
    m, p: if output='m, p'.
    z: if output='complex'.
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
    f, t, z = signal.stft(au, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    t = np.around(t, 2)
    if output == 'm':
        m = np.abs(z)
        return f, t, m
    elif output == 'm, p':
        m = np.abs(z)
        p = np.angle(z*np.exp((np.pi/2)*1.0j))
        return f, t, m, p
    elif output == 'complex':
        return f, t, z
    elif output == 'r, i':
        return f, t, z.real, z.imag
    else:
        raise ValueError('Parameter "output" has to be "m, p", "complex" or "r, i".')

def get_midi_frequency(midi):
    f = 440*np.exp2((midi-69)/12)
    return f

def 2f2cent(f1, f2):
    cent = 1200*np.log2(f2/f1)
    return cent

def ratio2cent(ratio):
    cent = 1200*np.log2(ratio)
    return cent

def cent2ratio(cent):
    ratio = np.exp2(cent/1200)
    return ratio

def get_pitch(au, sr, du, channel=0, given=True, given_pitch=None, given_cent=None):
    """
    Search around the given pitch within the freq to get the more exact pitch. For piano single note sound.

    Parameters:
    given: bool. Search with given pitch or not. This may be needed because pianos' lower notes may have some significant inharmonic frequencies caused by strings' logitudinal vibration.
    given_pitch: float (Hz). One possible pitch given, usually the standard pitch. 
    given_cent: folat (cent). Half of the cent band to search around the given pitch for the index of the frequency with maxinum amplitude.
    When given is True, both given_pitch and given_cent cannot be None.
    """
    if au.ndim == 1:
        pass
    elif au.ndim == 2:
        au = au[:, channel]
    else:
        raise ValueError('au.ndim needs to be 1 or 2.')
    f, t, m = stft(au, sr)
    m = m[:, 0].reshape(f.size)
    if given:
        if given_pitch != None and given_cent != None:
            f_l, f_h = given_pitch/given_cent, given_pitch*given_cent
            f_l_idx, f_h_idx = np.argmin(np.abs(f-f_l)), np.argmin(np.abs(f-f_h))
            f_cut, m_cut = f[f_l_idx: f_h_idx+1], m[f_l_idx: f_h_idx+1]
            pitch = f_cut[np.argmax(m_cut)]
            return pitch
        else:
            raise ValueError('When given is True, both given_pitch and given_cent cannot be None.')
    else:
        pitch = f[np.argmax(m)]
        return pitch

def plot_stft_m(f, t, m, win_idx=0):
    time = t[win_idx]
    x, y = f, m[:, win_idx]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('frequency magnitude')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_p(f, t, p, win_idx=0):
    time = t[win_idx]
    x, y = f, p[:, win_idx]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('frequency phase')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_z(f, t, z, win_idx=0):
    time = t[win_idx]
    z = z[:, win_idx]
    x, y1, y2 = f, z.real, z.imag
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, y1)
    ax2.plot(x, y2)
    fig.patch.set_facecolor('#D1DDC5')
    ax1.patch.set_facecolor('#D1DDC5')
    ax2.patch.set_facecolor('#D1DDC5')
    plt.xlabel('frquency')
    ax1.set_title(f'time = {time}s' + '\n' + 'real part')
    ax2.set_title('imaginary part')
    ax1.grid(color='grey', linewidth='1', linestyle='-.')
    ax2.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_r(f, t, r, win_idx=0):
    time = t[win_idx]
    x, y = f, r[:, win_idx]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('real part')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_i(f, t, i, win_idx=0):
    time = t[win_idx]
    x, y = f, i[:, win_idx]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('imaginary part')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_psd(f, Pxx):
    f = f.astype(np.int16)
    x, y = f, Pxx
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'Power Spectral Dessity')
    plt.xlabel('frquency')
    plt.ylabel('psd')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()
