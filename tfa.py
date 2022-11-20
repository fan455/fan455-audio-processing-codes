"""
Audio Time-Frequency Analysis codes based on numpy, scipy, resampy and matplotlib.
"""
import numpy as np
from scipy import signal
import resampy
from pitch import cent2ratio

def resample(au, sr, sr_new):
    return resampy.resample(au, sr, sr_new, axis=0)

def pitch_shift_cent(au, sr, cent):
    sr_stretch = int(np.rint(sr/cent2ratio(cent)))
    return resampy.resample(au, sr, sr_stretch, axis=0)

def pitch_shift_cent_2(au, sr, cent, sr_new):
    sr_stretch = int(np.rint(sr_new/cent2ratio(cent)))
    return resampy.resample(au, sr, sr_stretch, axis=0)

def pitch_shift_ratio(au, sr, ratio):
    sr_stretch = int(np.rint(sr/ratio))
    return resampy.resample(au, sr, sr_stretch, axis=0)

def pitch_shift_ratio_2(au, sr, ratio, sr_new):
    sr_stretch = int(np.rint(sr_new/ratio))
    return resampy.resample(au, sr, sr_stretch, axis=0)

def get_framed(au, sr, T=0.4, overlap=0.75, win='rectangular'):
    """
    Parameters
    au: ndarray. Needs to have mono shape (samples_num, ) or multi-channel shape (samples_num, channels_num)
    sr: float (Hz). Sample rate of input audio array.
    T: float (seconds). Time length of each window.
    overlap: float, proportion. Proportion of overlapping between windows.
    win: str. The window to apply to every frame.

    Returns
    au_f: ndarray. Framed audio with mono shape (window_num, samples) or multi-channel shape (window_num, samples_num, channels_num).
    """
    step, hop = int(sr*T), int(sr*T*(1-overlap))
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
        elif win == 'kaiser':
            au_f *= np.kaiser(step, 14).reshape((1, step, 1))
        elif win == 'hanning':
            au_f *= np.hanning(step).reshape((1, step, 1))
        elif win == 'hamming':
            au_f *= np.hamming(step).reshape((1, step, 1))
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
        elif win == 'kaiser':
            au_f *= np.kaiser(step, 14).reshape((1, step))
        elif win == 'hanning':
            au_f *= np.hanning(step).reshape((1, step))
        elif win == 'hamming':
            au_f *= np.hamming(step).reshape((1, step))
        else:
            raise ValueError(f'window "{win}" is not supported.')
        return au_f
    else:
        raise ValueError(f'au.ndim = {au.ndim} is not supported.')

def psd(au, sr, channel=None, T=1.0, overlap=0.5):
    if channel != None:
        if au.ndim == 2:
            au = au[:, channel]
        elif au.ndim == 1:
            pass
        else:
            raise ValueError('The input audio array has no dimension, or more than 2 dimensions which means it may be a framed audio.')
    f, Pxx = signal.welch(au, fs=sr, nperseg=int(sr*T), noverlap=int(sr*T*overlap), axis=0)
    return f, Pxx

class stft_class():

    def __init__(self, sr, T=0.05, overlap=0.75, win='hann'):
        """
        Only mono audio array is supported.

        Parameters:
        au: numpy.ndarray. Need to have 1 or 2 dimensions like normal single-channel or multi-channel audio. win_idxd audio needs to be converted to non-win_idxd audio first using other functions to have the right stft.
        sr: int. Sample rate of au.
        channel: int. If au has 2 dimensions, which channel to do stft. 0 represents the first channel. None will stft all the channels sepatately.
        output: str. 'm' will return a magnitudes array (zero or positive real values). 'm, p' will return two magnitudes and phases arrays. 'z' will return a complex array, the same as scipy. 'r, i' will return two real and imaginary arrays derived from the complex array.
        T: float (seconds). Time length of a each window. 
        overlap: float between 0 and 1. Overlap proportion between each two adjacent windows. 
        """
        self.sr, self.nperseg, self.noverlap, self.win = sr, int(sr*T), int(sr*T*overlap), win

    def forward(self, au, out_type='m'):
        """
        Returns:
        f: 1d array. As scipy.signal.stft returns.
        t: 1d array. As scipy.signal.stft returns.
        shape(f.size, t.size) array(s):
        m: if output='m'.
        m, p: if output='m, p'.
        z: if output='z'.
        z.real, z.imag: if output='z_r, z_i'.
        """
        f, t, z = signal.stft(au, fs=self.sr, nperseg=self.nperseg, noverlap=self.noverlap, window=self.win, axis=0)
        z = z.swapaxes(1, -1)
        print(f'z.shape = {z.shape}')
        if out_type == 'm':
            m = np.abs(z)
            return f, t, m
        elif out_type == 'm, p':
            m, p = np.abs(z), np.angle(z)
            #m, p = np.abs(z), np.angle(z*np.exp((np.pi/2)*1.0j))
            print(f'f.shape = {f.shape}')
            print(f't.shape = {t.shape}')
            print(f'm.shape = {m.shape}')
            print(f'p.shape = {p.shape}')
            return f, t, m, p
        elif out_type == 'z':
            return f, t, z
        elif out_type == 'z_r, z_i':
            return f, t, z.real, z.imag
        else:
            raise ValueError('Parameter "out_type" has to be "m", "m, p", "z" or "z_r, z_i".')

    def inverse(self, in_tup, in_type='m'):
        if in_type == 'm':
            m = in_tup
            del in_tup
            shape = m.shape
            p = np.unwrap(np.pi*np.random.uniform(-1, 1, shape), axis=1) # this uses random phase.
            z = np.empty(shape, dtype=np.complex128)
            z.real, z.imag = m*np.cos(p), m*np.sin(p)
        elif in_type == 'm, p':
            m, p = in_tup
            del in_tup
            shape = m.shape
            z = np.empty(shape, dtype=np.complex128)
            z.real, z.imag = m*np.cos(p), m*np.sin(p)
        elif in_type == 'z':
            z = in_tup
            del in_tup
        elif in_type == 'z_r, z_i':
            shape = in_tup[0].shape
            z = np.empty(shape, dtype=np.complex128)
            z.real, z.imag = in_tup
            del in_tup
        else:
            raise ValueError('Parameter "in_type" has to be "m", "m, p", "z" or "z_r, z_i".')
        t, au = signal.istft(z, fs=self.sr, nperseg=self.nperseg, noverlap=self.noverlap, window=self.win, time_axis=1, freq_axis=0)
        print(f'au.shape = {au.shape}')
        return au

    def re(self, au, re_type='m, p'):
        if re_type == 'm':
            f, t, m = self.forward(au, out_type=re_type)
            au_re = self.inverse(m, in_type=re_type)           
        elif re_type == 'm, p':
            f, t, m, p = self.forward(au, out_type=re_type)
            au_re = self.inverse((m, p), in_type=re_type)
        elif re_type == 'z':
            f, t, z = self.forward(au, out_type=re_type)
            au_re = self.inverse(z, in_type=re_type)
        elif re_type == 'z_r, z_i':
            f, t, z_r, z_i = self.forward(au, out_type=re_type)
            au_re = self.inverse((z_r, z_i), in_type=re_type)
        else:
            raise ValueError('Parameter "re_type" has to be "m", "m, p", "z" or "z_r, z_i".')
        return au_re
    
def stft(au, sr, T=0.1, overlap=0.75, output='m'):

    f, t, z = signal.stft(au, fs=sr, nperseg=int(sr*T), noverlap=int(sr*T*overlap))
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

def istft(sr, m, p=None, T=0.1, overlap=0.75):
    """
    Only mono audio array is supported.
    """
    shape = m.shape
    if p == None:
        p = np.unwrap(np.random.uniform(0, 2*np.pi, shape))
    tanp = np.tan(p)
    a = m/np.sqrt(1+np.square(tanp))
    b = a*tanp
    del tanp
    z = np.empty(shape, dtype=np.complex128)
    z.real, z.imag = a, b
    del a, b
    return signal.istft(z, fs=sr, nperseg=int(sr*T), noverlap=int(sr*T*overlap))

def get_sinewave(f, phase=0, A=1, du=1, sr=48000, stereo=True, ls=None, ts=None):
    """
    Generate a pure sine wave for loudness testing.
    f: float (Hz). Frequency.
    phase: float (rad angle). Initial phase.
    A: float (amp). Maxinum amplitude.
    du: float (seconds). Duration of sinewave.
    ls: float (seconds). Duration of leading silence.
    ts: float (seconds). Duration of trailing silence.
    """
    t = np.arange(0, int(du*sr))/sr
    y = A*np.sin(2*np.pi*f*t + phase)
    if ls:
        y = np.append(np.zeros(int(ls*sr)), y)
    if ts:
        y = np.append(y, np.zeros(int(ts*sr)))
    size = y.size
    if stereo:
        return np.broadcast_to(y.reshape((size, 1)), (size, 2))
    else:
        return y
    
def get_white_noise(sr, du, A=0.5, ls=None, ts=None, stereo=False):
    size = int(sr*du)
    if stereo == False: # mono
        noise = A*np.random.uniform(-1, 1, size)
        if ls:
            noise = np.append(np.zeros(int(sr*ls)), noise)
        if ts:
            noise = np.append(noise, np.zeros(int(sr*ts)))
    else:
        noise = A*np.random.uniform(-1, 1, 2*size).reshape((size, 2))
        if ls:
            noise = np.append(np.zeros((int(sr*ls), 2)), noise, axis=0)
        if ts:
            noise = np.append(noise, np.zeros((int(sr*ts), 2)), axis=0)
    return noise

def get_idx_array(y):
    return np.arange(y.size)

def sample2time(n, sr):
    # start from 0
    return n/sr

def time2sample(t, sr):
    # start from 0
    return (sr*t).astype(np.int64)

def get_pitch_given(au, sr, channel=0, du=None, given_freq=440, given_cent=100, cent_step=1):
    """
    Detect the pitch of audio (specifically piano single note) given a pitch, cent band and cent step, using discrete time fourier transform in limited frequency range.
    The computation will be quite slow since it does not use FFT, but it's much more accurate than scipy.signal.stft in terms of frequency resolution. 
    I've ensured the cpu and memory pressure won't be high by using for-loop.
    
    Parameters:
    au: ndarray (float between -1 and 1). The input audio.
    sr: int. Sample rate of audio.
    channel: int. The index of the audio channel to analyze. Only supports 1-channel analysis. None (using all channels) is not supported.
    du: None or float (seconds). The duration of audio to be analyzed. If set to None, it will be the maxinum integar seconds available.
    given_freq: float (Hz).
    given_cent: positive float (cent). Half of the cent band around the given frequency for pitch detection.
    cent_step: float (cent). The distance between Fourier transform's frequencies measured in cents, i.e. the resolution of frequencies.
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
