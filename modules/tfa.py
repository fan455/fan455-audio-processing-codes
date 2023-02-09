# Audio Time-Frequency Analysis

import numpy as np
from scipy import fft, signal

# discrete cosine transform
def dct_z(y, axis=-1, dct_type=2):
    return fft.dct(y, dct_type, axis=axis, norm='backward')

def dct_zf(y, sr, axis=-1, dct_type=2):
    z = fft.dct(y, dct_type, axis=axis, norm='backward')
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return z, f

def dct_mf(y, sr, axis=-1, dct_type=2):
    # returns: magnitude, frequency
    z = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(z)
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, f

def dct_mf(y, sr, axis=-1, dct_type=2):
    # returns: magnitude, frequency
    z = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(z)
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, f

def dct_mp(y, axis=-1, dct_type=2):
    # returns: magnitude, sign
    z = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.sign(z)
    return m, p

def dct_mpf(y, sr, axis=-1, dct_type=2):
    # returns: magnitude, sign, frequency
    z = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.sign(z)
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, p, f

def idct_z(z, axis=-1, dct_type=2):
    return fft.idct(z, dct_type, axis=axis, norm='backward')

def idct_mp(m, p, axis=-1, dct_type=2):
    return fft.idct(m*p, dct_type, axis=axis, norm='backward')

# real discrete Fourier transform
def rfft_z(y, axis=-1):
    return fft.rfft(y, axis=axis, norm='backward')

def rfft_zf(y, sr, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return z, f

def rfft_m(y, axis=-1):
    # This returns frequency magnitudes (real positive numbers) instead of complex numbers.
    return np.abs(fft.rfft(y, axis=axis, norm='backward'))

def rfft_mf(y, sr, axis=-1):
    # This returns frequency magnitudes and the frequencies consistent with sr.
    m = np.abs(fft.rfft(y, axis=axis, norm='backward'))
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return m, f

def rfft_mp(y, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    return m, p

def rfft_mpf(y, sr, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return m, p, f

def irfft_z(z, time_size_is_even=True, axis=-1):
    if time_size_is_even:
        return fft.irfft(z, axis=axis, norm='backward')
    else:
        return fft.irfft(z, n=z.shape[axis]*2-1, axis=axis, norm='backward')

def irfft_mp(m, p, time_size_is_even=True, axis=-1):
    z = np.empty(m.shape, dtype=np.complex128)
    z.real, z.imag = m*np.cos(p), m*np.sin(p)
    if time_size_is_even:
        return fft.irfft(z, axis=axis, norm='backward')
    else:
        return fft.irfft(z, n=z.shape[axis]*2-1, axis=axis, norm='backward')

# discrete Fourier transform
def fft_z(y, axis=-1):
    return fft.fft(y, axis=axis, norm='backward')

def fft_zf(y, sr, axis=-1):
    z = fft.fft(y, axis=axis, norm='backward')
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return z, f

def fft_m(y, axis=-1):
    # This returns frequency magnitudes (real positive numbers) instead of complex numbers.
    return np.abs(fft.fft(y, axis=axis, norm='backward'))

def fft_mf(y, sr, axis=-1):
    # This returns frequency magnitudes and the frequencies consistent with sr.
    m = np.abs(fft.fft(y, axis=axis, norm='backward'))
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, f

def fft_mp(y, axis=-1):
    z = fft.fft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    return m, p

def fft_mpf(y, sr, axis=-1):
    z = fft.fft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, p, f

def ifft_z(z, axis=-1):
    return fft.ifft(z, axis=axis, norm='backward')

def ifft_mp(m, p, axis=-1):
    z = np.empty(m.shape, dtype=np.complex128)
    z.real, z.imag = m*np.cos(p), m*np.sin(p)
    return fft.ifft(z, axis=axis, norm='backward')

# Hilbert transform
def hilbert_z(y, axis=-1):
    # This returns the analytic signal of y.
    return signal.hilbert(y, axis=axis)

def hilbert_ap(y, axis=-1):
    # This returns (am, pm), the instaneous amplitude and phase arrays.
    z = signal.hilbert(y, axis=axis)
    return np.abs(z), np.unwrap(np.angle(z))

def hilbert_af(y, sr, axis=-1):
    # This returns (am, fm), the instaneous amplitude and frequency arrays.
    z = signal.hilbert(y, axis=axis)
    am = np.abs(z)
    fm = 0.5*sr*np.diff(np.unwrap(np.angle(z)), axis=axis)/np.pi
    return am, fm

def hilbert_apf(y, sr, axis=-1):
    # This returns (am, pm, fm).
    z = signal.hilbert(y, axis=axis)
    am = np.abs(z)
    pm = np.unwrap(np.angle(z))
    fm = 0.5*sr*np.diff(pm, axis=axis)/np.pi
    return am, pm, fm

def ihilbert_z(z):
    return np.real(z)

def ihilbert_ap(am, pm):
    return am*np.cos(pm)

# generate test signal
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

# pitch detection
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
