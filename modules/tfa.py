# Audio Time-Frequency Analysis
# recommended import line: from modules.tfa import *

import numpy as np
from scipy import fft, signal

# discrete cosine transform
def get_dct(y, axis=-1, dct_type=2):
    return fft.dct(y, dct_type, axis=axis, norm='backward')

def get_dctmf(y, sr, axis=-1, dct_type=2):
    # returns: magnitude, frequency
    y_dct = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(y_dct)
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, f

def get_dctmp(y, axis=-1, dct_type=2):
    # returns: magnitude, sign
    y_dct = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(y_dct)
    p = np.sign(y_dct)
    return m, p

def get_dctmpf(y, sr, axis=-1, dct_type=2):
    # returns: magnitude, sign, frequency
    y_dct = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(y_dct)
    p = np.sign(y_dct)
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, p, f

def get_idct(y_dct, axis=-1, dct_type=2):
    return fft.idct(y_dct, dct_type, axis=axis, norm='backward')

def get_idctmp(m, p, axis=-1, dct_type=2):
    return fft.idct(m*p, dct_type, axis=axis, norm='backward')

# real discrete Fourier transform
def get_rfft(y, axis=-1):
    return fft.rfft(y, axis=axis, norm='backward')

def get_rfftm(y, axis=-1):
    # This returns frequency magnitudes (real positive numbers) instead of complex numbers.
    return np.abs(fft.rfft(y, axis=axis, norm='backward'))

def get_rfftmf(y, sr, axis=-1):
    # This returns frequency magnitudes and the frequencies consistent with sr.
    m = np.abs(fft.rfft(y, axis=axis, norm='backward'))
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return m, f

def get_rfftmp(y, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    return m, p

def get_rfftmpf(y, sr, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return m, p, f

def get_irfft(z, time_size_is_even=True, axis=-1):
    if time_size_is_even:
        return fft.irfft(z, axis=axis, norm='backward')
    else:
        return fft.irfft(z, n=z.shape[axis]*2 - 1, axis=axis, norm='backward')

def get_irfftmp(m, p, time_size_is_even=True, axis=-1):
    z = np.empty(m.shape, dtype=np.complex128)
    z.real, z.imag = m*np.cos(p), m*np.sin(p)
    if time_size_is_even:
        return fft.irfft(z, axis=axis, norm='backward')
    else:
        return fft.irfft(z, n=z.shape[axis]*2 - 1, axis=axis, norm='backward')

# discrete Fourier transform
def get_fft(y, axis=-1):
    return fft.fft(y, axis=axis, norm='backward')

def get_fftm(y, axis=-1):
    # This returns frequency magnitudes (real positive numbers) instead of complex numbers.
    return np.abs(fft.fft(y, axis=axis, norm='backward'))

def get_fftmf(y, sr, axis=-1):
    # This returns frequency magnitudes and the frequencies consistent with sr.
    m = np.abs(fft.fft(y, axis=axis, norm='backward'))
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, f

def get_fftmp(y, axis=-1):
    z = fft.fft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    return m, p

def get_fftmpf(y, sr, axis=-1):
    z = fft.fft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, p, f

def get_ifft(z, axis=-1):
    return fft.ifft(z, axis=axis, norm='backward')

def get_ifftmp(m, p, axis=-1):
    z = np.empty(m.shape, dtype=np.complex128)
    z.real, z.imag = m*np.cos(p), m*np.sin(p)
    return fft.ifft(z, axis=axis, norm='backward')

# Hilbert transform
def get_hilbert(y, axis=-1):
    return signal.hilbert(y, axis=axis)

def get_hilbert_ap(y, axis=-1):
    # This returns (am, pm), the instaneous amplitude and phase arrays.
    ya = signal.hilbert(y, axis=axis)
    return np.abs(ya), np.unwrap(np.angle(ya))

def get_hilbert_af(y, sr, axis=-1):
    # This returns (am, fm), the instaneous amplitude and frequency arrays.
    # The length of the fm array is reduced by one.
    ya = signal.hilbert(y, axis=axis)
    am = np.abs(ya)
    if y.ndim == 2:
        if axis == -1 or axis == 1:
            fm = 0.5*sr*(ya.real[:,:-1]*np.diff(ya.imag,axis=1) - \
                         ya.imag[:,:-1]*np.diff(ya.real,axis=1)) / \
                         ((ya.real[:,:-1]**2 + ya.imag[:,:-1]**2)*np.pi)
        elif axis == 0:
            fm = 0.5*sr*(ya.real[:-1,:]*np.diff(ya.imag,axis=0) - \
                         ya.imag[:-1,:]*np.diff(ya.real,axis=0)) / \
                         ((ya.real[:-1,:]**2 + ya.imag[:-1,:]**2)*np.pi)            
    elif y.ndim == 1:
        fm = 0.5*sr*(ya.real[:-1]*np.diff(ya.imag) - ya.imag[:-1]*np.diff(ya.real)) / \
             ((ya.real[:-1]**2 + ya.imag[:-1]**2)*np.pi)
    return am, fm

def get_ihilbert(ya):
    return np.real(ya)

def get_ihilbert_ap(am, pm):
    return am*np.cos(pm) 

# Test signal
def get_sinewave(sr, du=1.0, f=440, phase=0, A=0.3, stereo=False, ls=None, ts=None):
    """
    Generate a pure sine wave for testing.
    sr: positive int (Hz). Sample rate.
    du: positive float (seconds). Duration of sinewave.
    f: positive float (Hz). Frequency.
    phase: float (rad angle). Initial phase.
    A: positive float (amp). Maxinum amplitude.
    ls: positive float (seconds). Duration of leading silence.
    ts: positive float (seconds). Duration of trailing silence.
    stereo: bool. If true, return a 2d array. If false, return a 1d array.
    """
    size = int(sr*du)
    t = np.arange(0, size)/sr
    y = A*np.sin(2*np.pi*f*t + phase)
    if ls:
        y = np.append(np.zeros(int(ls*sr)), y)
    if ts:
        y = np.append(y, np.zeros(int(ts*sr)))
    if stereo:
        return np.broadcast_to(y.reshape((size, 1)), (size, 2))
    else:
        return y
    
def get_uniform_noise(sr, du=1.0, A=0.3, ls=None, ts=None, stereo=False):
    """
    Generate a uniform white noise signal for testing.
    sr: positive int (Hz). Sample rate.
    du: positive float (seconds). Duration of sinewave.
    A: positive float (amp). Maxinum amplitude.
    ls: positive float (seconds). Duration of leading silence.
    ts: positive float (seconds). Duration of trailing silence.
    stereo: bool. If true, return a 2d array. If false, return a 1d array.
    """
    size = int(sr*du)
    if stereo == False: # mono
        noise = A*np.random.uniform(-1, 1, size)
        if ls:
            noise = np.append(np.zeros(int(sr*ls)), noise)
        if ts:
            noise = np.append(noise, np.zeros(int(sr*ts)))
    else:
        noise = A*np.random.uniform(-1, 1, (size, 2))    
        if ls:
            noise = np.append(np.zeros((int(sr*ls), 2)), noise, axis=0)
        if ts:
            noise = np.append(noise, np.zeros((int(sr*ts), 2)), axis=0)
    return noise

def get_gaussian_noise(sr, du=1.0, A=0.3, limit=3.0, ls=None, ts=None, stereo=False):
    """
    Generate a gaussian white noise signal for testing.
    sr: positive int (Hz). Sample rate.
    du: positive float (seconds). Duration of sinewave.
    A: positive float (amp). Maxinum amplitude.
    limit: positive float. Values out of range(-limit*std, limit*std) will be set to -limit*std or limit*std.
    ls: positive float (seconds). Duration of leading silence.
    ts: positive float (seconds). Duration of trailing silence.
    stereo: bool. If true, return a 2d array. If false, return a 1d array.
    """
    size = int(sr*du)
    if stereo == False: # mono
        noise = np.random.normal(0.0, 1.0, size)*A/limit
        noise[noise < -A] = -A
        noise[noise > A] = A
        if ls:
            noise = np.append(np.zeros(int(sr*ls)), noise)
        if ts:
            noise = np.append(noise, np.zeros(int(sr*ts)))
    else:
        noise = np.random.normal(0.0, 1.0, (size, 2))*A/limit
        noise[noise < -A] = -A
        noise[noise > A] = A
        if ls:
            noise = np.append(np.zeros((int(sr*ls), 2)), noise, axis=0)
        if ts:
            noise = np.append(noise, np.zeros((int(sr*ts), 2)), axis=0)
    return noise

def get_silence(sr, du=1.0, stereo=False):
    """
    Generate a silence signal for testing.
    sr: int (Hz). Sample rate.
    du: float (seconds). Duration of sinewave.
    stereo: bool. If true, return a 2d array. If false, return a 1d array.
    """
    size = int(sr*du)
    if stereo == False:
        return np.zeros(size)
    else:
        return np.zeros((size, 2))

# Others
def get_pitch_given(au, sr, du=None, given_freq=440, given_cent=100, cent_step=1):
    """
    Detect the pitch of audio (specifically piano single note) given a pitch, cent band and cent step, using discrete time fourier transform in limited frequency range.
    The computation will be quite slow since it does not use FFT, but it's much more accurate than signal.stft in terms of frequency resolution. 
    I've ensured the cpu and memory pressure won't be high by using for-loop.
    
    Parameters:
    au: ndarray (float between -1 and 1). The input audio.
    sr: int. Sample rate of audio.
    du: None or float (seconds). The duration of audio to be analyzed. If set to None, it will be the maxinum integar seconds available.
    given_freq: float (Hz). The central frequency around which pitch will be detected.
    given_cent: positive float (cent). Half of the cent band around the given frequency for pitch detection.
    cent_step: float (cent). The distance between Fourier transform's frequencies measured in cents, i.e. the resolution of frequencies.
    """
    if au.ndim == 1:
        pass
    elif au.ndim == 2:
        au = np.average(au, axis=-1)
    else:
        raise ValueError('The input audio array has no dimension, or over 2 dimensions which means it may be a framed audio.')
    if du is None:
        size = au.size
    else:
        size = int(sr*du)
        au = au[0: size]
    t = np.arange(0, size)/sr
    F = given_freq*np.exp2(np.arange(-given_cent, given_cent+1, cent_step)/1200)
    M = np.empty(0)
    for f in F:
        m = np.abs(np.average(au*np.exp(-2*np.pi*f*t*1.0j)))
        M = np.append(M, m)
    pitch = F[np.argmax(M)]
    print(f'{round(pitch, 2)}Hz is the detected pitch given {round(given_freq, 2)}Hz, {round(given_cent, 2)} cent band and {np.round(cent_step, 2)} cent step.')
    return pitch
