"""
Audio Time-Frequency Analysis codes based on numpy, scipy, resampy and matplotlib.
"""
import numpy as np
from scipy import fft, signal
import resampy
from pitch import cent2ratio
from loudness import amp2db, db2amp

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

class fft_class():

    def __init__(self, sr=None, _type='m, p', random_phase_type='mono'):
        self.sr, self._type, self.random_phase_type = sr, _type, random_phase_type

    def fw(self, au):
        z = fft.rfft(au, axis=0, norm='forward')
        print(f'au.shape = {au.shape}')
        print(f'z.shape = {z.shape}')
        if self._type == 'm':
            m = np.abs(z)
            return m
        elif self._type == 'm, p':
            m, p = np.abs(z), np.angle(z*np.exp(0.5*np.pi*1.0j))
            return m, p
        elif self._type == 'z':
            return z
        elif self._type == 'z.real, z.imag':
            return z.real, z.imag
        else:
            raise ValueError('Parameter self._type has to be "m", "m, p", "z" or "z.real, z.imag".')

    def bw(self, in_tup, nsample=None):
        if self._type == 'm':
            m = in_tup
            del in_tup
            p = self.get_random_phase(nsample, m.ndim)
            z = np.empty(m.shape, dtype=np.complex128)
            z.real, z.imag = m*np.cos(p), m*np.sin(p)
        elif self._type == 'm, p':
            m, p = in_tup
            del in_tup
            p -= 0.5*np.pi
            z = np.empty(m.shape, dtype=np.complex128)
            z.real, z.imag = m*np.cos(p), m*np.sin(p)
        elif self._type == 'z':
            z = in_tup
            del in_tup
        elif self._type == 'z.real, z.imag':
            z = np.empty(in_tup[0].shape, dtype=np.complex128)
            z.real, z.imag = in_tup
            del in_tup
        else:
            raise ValueError('Parameter self._type has to be "m", "m, p", "z" or "z.real, z.imag".')
        au_re = fft.irfft(z, axis=0, norm='forward')
        print(f'au_re.shape = {au_re.shape}')
        return au_re

    def re(self, au):
        if self._type == 'm':
            m = self.fw(au)
            nsample = au.shape[0]
            au_re = self.bw(m, nsample)           
        elif self._type == 'm, p':
            m, p = self.fw(au)
            au_re = self.bw((m, p))
        elif self._type == 'z':
            z = self.fw(au)
            au_re = self.bw(z)
        elif self._type == 'z.real, z.imag':
            z_r, z_i = self.fw(au)
            au_re = self.bw((z_r, z_i))
        else:
            raise ValueError('Parameter self._type has to be "m", "m, p", "z" or "z.real, z.imag".')
        return au_re 

    def get_random_phase(self, nsample, m_ndim):
        if m_ndim == 2:
            if self.random_phase_type == 'mono':
                noise = 0.5*np.random.uniform(-1, 1, nsample)
                noise = np.stack((noise, noise), axis=-1)
            elif self.random_phase_type == 'stereo':
                noise = 0.5*np.random.uniform(-1, 1, (nsample, 2))
            else:
                raise ValueError('self.random_phase_type != "mono" or "stereo"')
        elif m_ndim == 1:
            noise = 0.5*np.random.uniform(-1, 1, nsample)
        else:
            raise ValueError('m_ndim != 1 or 2')
        z_noise = fft.rfft(noise, axis=0, norm='forward')
        p_noise = np.angle(z_noise*np.exp(0.5*np.pi*1.0j))
        print(f'noise.shape = {noise.shape}')
        print(f'nsample = {nsample}')
        print(f'p_noise.shape = {p_noise.shape}')
        return p_noise

    def re_compare(self, au, au_re):
        print('reconstruction comparison:')
        print(f'max error: {round(amp2db(np.amax(np.abs(au_re[:au.shape[0], :] - au))), 4)}db')
        if self.sr:
            print(f'difference in length: {round((au_re.shape[0] - au.shape[0])/self.sr, 4)} seconds')
            
class stft_class():

    def __init__(self, sr, T=0.1, overlap=0.75, fft_ratio=1.0, win='blackmanharris', _type='m, p', random_phase_type='mono'):
        """
        Parameters:
        sr: int (Hz). Sample rate, ususally 44100 or 48000.
        T: float (seconds). Time length of a each window. 
        overlap: float (ratio between 0 and 1). Overlap ratio between each two adjacent windows.
        fft_ratio: float (ratio >= 1). The fft ratio relative to T.
        win: str. Please refer to scipy's window functions. Window functions like kaiser will require a tuple input including additional parameters. e.g. ('kaiser', 14.0)
        _type: str ('m', 'm, p', 'z' or 'z.real, z.imag'). Please refer to the illustration of the returns of self.forward().
        random_phase_type: str. Only useful when _type='m' and audio is stereo. If it's 'mono'('stereo'), stereo reconstruction will use the same (different) random phases for both channels.
        """
        self.sr, self.nperseg, self.noverlap, self.nfft = sr, int(sr*T), int(sr*T*overlap), int(sr*T*fft_ratio)
        self.nhop = self.nperseg - self.noverlap
        self.win, self._type = signal.windows.get_window(win, self.nperseg, fftbins=True), _type
        self.random_phase_type = random_phase_type

    def fw(self, au):
        """
        Short-Time Fourier Transform

        Parameters:
        au: ndarray (dtype = float between -1 and 1). Need to have 1 or 2 dimensions like normal single-channel or multi-channel audio. 

        Returns:
        f: 1d array. As scipy.signal.stft returns.
        t: 1d array. As scipy.signal.stft returns.
        m: if self._type='m'. The magnitudes array of shape (f.size, t.size) or (f.size, t.size, au.shape[-1]). PLEASE NOTE that the istft will use phases of a white noise!
        m, p: if self._type='m, p'. The magnitudes array and phases array of shapes (f.size, t.size) or (f.size, t.size, au.shape[-1]). The phase range is [-pi, pi].
        z: if self._type='z'. The complex array of shape (f.size, t.size) or (f.size, t.size, au.shape[-1]).
        z.real, z.imag: if self._type='z.real, z.imag'. The complex array' real array and imaginary array of shapes (f.size, t.size) or (f.size, t.size, au.shape[-1]).
        """
        f, t, z = signal.stft(au, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, axis=0)
        z = z.swapaxes(1, -1)
        print(f'au.shape = {au.shape}')
        print(f'f.shape = {f.shape}')
        print(f't.shape = {t.shape}')
        print(f'z.shape = {z.shape}')
        if self._type == 'm':
            m = np.abs(z)
            print(f'm.shape = {m.shape}')
            return f, t, m
        elif self._type == 'm, p':
            m, p = np.abs(z), np.angle(z*np.exp(0.5*np.pi*1.0j))
            print(f'm.shape = {m.shape}')
            print(f'p.shape = {p.shape}')
            return f, t, m, p
        elif self._type == 'z':
            return f, t, z
        elif self._type == 'z.real, z.imag':
            return f, t, z.real, z.imag
        else:
            raise ValueError('Parameter self._type has to be "m", "m, p", "z" or "z.real, z.imag".')

    def bw(self, in_tup, nsample=None):
        """
        Inverse Short-Time Fourier Transform

        Parameters:
        in_tup: an ndarray or a tuple containing 2 ndarrays corresponding to self._type. Please refer to the illustration of the returns of self.forward().
        
        Returns:
        au_re: ndarray. Audio array after inverse short-time fourier transform.
        """
        if self._type == 'm':
            m = in_tup
            del in_tup
            p = self.get_random_phase(nsample, m.ndim)
            z = np.empty(m.shape, dtype=np.complex128)
            z.real, z.imag = m*np.cos(p), m*np.sin(p)
        elif self._type == 'm, p':
            m, p = in_tup
            del in_tup
            p -= 0.5*np.pi
            z = np.empty(m.shape, dtype=np.complex128)
            z.real, z.imag = m*np.cos(p), m*np.sin(p)
        elif self._type == 'z':
            z = in_tup
            del in_tup
        elif self._type == 'z.real, z.imag':
            z = np.empty(in_tup[0].shape, dtype=np.complex128)
            z.real, z.imag = in_tup
            del in_tup
        else:
            raise ValueError('Parameter self._type has to be "m", "m, p", "z" or "z.real, z.imag".')
        t, au_re = signal.istft(z, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, time_axis=1, freq_axis=0)
        print(f'au_re.shape = {au_re.shape}')
        return au_re

    def re(self, au):
        """
        Reconstruct an audio array using stft and then istft. Please refer to the illustration of the returns of self.forward().

        Parameters:
        au: ndarray (dtype = float between -1 and 1). Need to have 1 or 2 dimensions like normal single-channel or multi-channel audio. 
        """
        if self._type == 'm':
            nsample = au.shape[0]
            f, t, m = self.fw(au)
            au_re = self.bw(m, nsample)           
        elif self._type == 'm, p':
            f, t, m, p = self.fw(au)
            au_re = self.bw((m, p))
        elif self._type == 'z':
            f, t, z = self.fw(au)
            au_re = self.bw(z)
        elif self._type == 'z.real, z.imag':
            f, t, z_r, z_i = self.fw(au)
            au_re = self.bw((z_r, z_i))
        else:
            raise ValueError('Parameter self._type has to be "m", "m, p", "z" or "z.real, z.imag".')
        return au_re        

    def get_random_phase(self, nsample, m_ndim):
        if m_ndim == 3:
            if self.random_phase_type == 'mono':
                noise = 0.5*np.random.uniform(-1, 1, nsample)
                noise = np.stack((noise, noise), axis=-1)
            elif self.random_phase_type == 'stereo':
                noise = 0.5*np.random.uniform(-1, 1, (nsample, 2))
            else:
                raise ValueError('self.random_phase_type != "mono" or "stereo"')
        elif m_ndim == 2:
            noise = 0.5*np.random.uniform(-1, 1, nsample)
        else:
            raise ValueError('m_ndim != 2 or 3')
        f_noise, t_noise, z_noise = signal.stft(noise, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, axis=0)
        z_noise = z_noise.swapaxes(1, -1)
        p_noise = np.angle(z_noise*np.exp(0.5*np.pi*1.0j))
        print(f'nsample = {nsample}')
        print(f'p_noise.shape = {p_noise.shape}')
        return p_noise

    def re_compare(self, au, au_re):
        print('reconstruction comparison:')
        print(f'max error: {round(amp2db(np.amax(np.abs(au_re[:au.shape[0], :] - au))), 4)}db')
        print(f'difference in length: {round((au_re.shape[0] - au.shape[0])/self.sr, 4)} seconds')

def get_sinewave(sr, du, f, phase=0, A=0.5, stereo=False, ls=None, ts=None):
    """
    Generate a pure sine wave for testing.
    sr: int (Hz). Sample rate.
    du: float (seconds). Duration of sinewave.
    f: float (Hz). Frequency.
    phase: float (rad angle). Initial phase.
    A: float (amp). Maxinum amplitude.
    ls: float (seconds). Duration of leading silence.
    ts: float (seconds). Duration of trailing silence.
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
    
def get_white_noise(sr, du, A=0.5, win=None, ls=None, ts=None, stereo=False):
    """
    Generate a uniform white noise signal for testing.
    sr: int (Hz). Sample rate.
    du: float (seconds). Duration of sinewave.
    A: float (amp). Maxinum amplitude.
    ls: float (seconds). Duration of leading silence.
    ts: float (seconds). Duration of trailing silence.
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

def get_silence(sr, du, stereo=False):
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

def get_framed(au, sr, T=0.4, overlap=0.75, win='hamming'):
    """
    Parameters
    au: ndarray. Needs to have mono shape (samples_num, ) or multi-channel shape (samples_num, channels_num)
    sr: float (Hz). Sample rate of input audio array.
    T: float (seconds). Time length of each window.
    overlap: float, proportion. Proportion of overlapping between windows.
    win: str or tuple. The window to apply to every frame. No need to provide window size. Please refer to scipy.signal.get_windows.

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
        if win:
            au_f *= signal.get_window(win, step).reshape((1, step, 1))
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
        if win:
            au_f *= signal.get_window(win, step).reshape((1, step))
        return au_f
    else:
        raise ValueError(f'au.ndim = {au.ndim} is not supported.')
