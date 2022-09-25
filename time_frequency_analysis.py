"""
Audio Time-Frequency Analysis codes based on numpy, scipy and matplotlib.
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def stft(au, sr, channel=0, output='m', nperseg=None, noverlap=0):
    """
    Parameters:
    au: numpy.ndarray. Need to have 1 or 2 dimensions like normal single-channel or multi-channel audio. Framed audio needs to be converted to non-framed audio first using other functions to have the right stft.
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
    f = f.astype(np.int16)
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

def plot_stft_m(f, t, m, frame=0):
    time = t[frame]
    x, y = f, m[:, frame]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('frequency magnitude')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_p(f, t, p, frame=0):
    time = t[frame]
    x, y = f, p[:, frame]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('frequency phase')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_z(f, t, z, frame=0):
    time = t[frame]
    z = z[:, frame]
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

def plot_stft_r(f, t, r, frame=0):
    time = t[frame]
    x, y = f, r[:, frame]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('real part')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()

def plot_stft_i(f, t, i, frame=0):
    time = t[frame]
    x, y = f, i[:, frame]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#D1DDC5')
    ax.patch.set_facecolor('#D1DDC5')
    ax.plot(x, y)
    ax.set_title(f'time = {time}s')
    plt.xlabel('frquency')
    plt.ylabel('imaginary part')
    ax.grid(color='grey', linewidth='1', linestyle='-.')
    plt.show()
