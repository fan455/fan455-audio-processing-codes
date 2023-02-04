"""
Audio Time-Frequency Analysis
"""
import numpy as np
from scipy import fft, signal

# IIR Filter
def get_iir_fr(b, a, sr, nf=None):
    # get the frequency response of a digital iir filter
    f, z = signal.freqz(b, a, fs=sr)
    return f, 20*np.log10(abs(z)), np.unwrap(np.angle(z)) # f, amp, phase

def get_iirsos_fr(sos, sr, nf=None):
    # get the frequency response of a digital sos iir filter
    if sos.ndim == 1:
        sos = sos.reshape((1, 6))
    f, z = signal.sosfreqz(sos, fs=sr)
    return f, 20*np.log10(abs(z)), np.unwrap(np.angle(z)) # f, amp, phase

def get_iirsos(y, sos, axis=0):
    if sos.ndim == 1:
        sos = sos.reshape((1, 6))
    return signal.sosfilt(sos, y, axis=axis)

# Equalizer
def get_eq_sos(sr, f0, dBgain, Q, eq_type):
    # get the sos parameters for eq.
    w0 = 2*np.pi*f0/sr
    cosw0, sinw0 = np.cos(w0), np.sin(w0)
    alpha = 0.5*sinw0/Q
    if eq_type == 'low pass':
        a0_ = 1+alpha
        b = np.array([0.5*(1-cosw0), 1-cosw0, 0.5*(1-cosw0)])/a0_
        a = np.array([1.0, -2*cosw0/a0_, (1-alpha)/a0_])
    elif eq_type == 'high pass':
        a0_ = 1+alpha
        b = np.array([0.5*(1+cosw0), -1-cosw0, 0.5*(1+cosw0)])/a0_
        a = np.array([1.0, -2*cosw0/a0_, (1-alpha)/a0_])
    elif eq_type == 'peak':
        A = np.power(10, dBgain/40)
        a0_ = 1+alpha/A
        b = np.array([1+alpha*A, -2*cosw0, 1-alpha*A])/a0_
        a = np.array([1.0, -2*cosw0/a0_, (1-alpha/A)/a0_])
    elif eq_type == 'band pass':
        a0_ = 1+alpha
        b = np.array([alpha, 0.0, -alpha])/a0_
        a = np.array([1.0, -2*cosw0/a0_, (1-alpha)/a0_])
    elif eq_type == 'band pass QdB':
        a0_ = 1+alpha
        b = np.array([Q*alpha, 0.0, -Q*alpha])/a0_
        a = np.array([1.0, -2*cosw0/a0_, (1-alpha)/a0_])
    elif eq_type == 'low shelf':
        A = np.power(10, dBgain/40)
        a0_ = (A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha
        b0 = A*((A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
        b1 = 2*A*((A-1) - (A+1)*cosw0)
        b2 = A*((A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
        a0 = 1.0
        a1 = -2*((A-1) + (A+1)*cosw0)
        a2 = (A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha
        b = A*np.array([b0, b1, b2])/a0_
        a = A*np.array([a0, a1/a0_, a2/a0_])
    elif eq_type == 'low shelf':
        a0_ = (A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha
        A = np.power(10, dBgain/40)
        b0 = A*((A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
        b1 = -2*A*((A-1) + (A+1)*cosw0)
        b2 = A*((A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
        a0 = 1.0
        a1 = 2*((A-1) - (A+1)*cosw0)
        a2 = (A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha
        b = A*np.array([b0, b1, b2])/a0_
        a = A*np.array([a0, a1/a0_, a2/a0_])
    elif eq_type == 'notch':
        a0_ = 1+alpha
        b = np.array([1.0, -2*cosw0, 1.0])/a0_
        a = np.array([1.0, -2*cosw0/a0_, (1-alpha)/a0_])
    elif eq_type == 'all pass':
        a0_ = 1+alpha
        b = np.array([1-alpha, -2*cosw0, 1+alpha])/a0_
        a = np.array([1.0, -2*cosw0/a0_, (1-alpha)/a0_])
    else:
        raise ValueError(f'eq_type "{eq_type}" is not supported.')
    return np.append(b, a)

def get_eq_fr(sr, f0, dBgain, Q, eq_type):
    sos = get_eq_sos(sr, f0, dBgain, Q, eq_type)
    sos = sos.reshape((1, 6))
    f, z = signal.sosfreqz(sos, fs=sr)
    return f, 20*np.log10(abs(z)), np.unwrap(np.angle(z)) # f, amp, phase

def get_eq(y, sr, f0, dBgain, Q, eq_type, axis=0):
    sos = get_eq_sos(sr, f0, dBgain, Q, eq_type)
    sos = sos.reshape((1, 6))
    return signal.sosfilt(sos, y, axis=axis)

def get_eqsos_sos(sr, f0: list, dBgain: list, Q: list, eq_type: list):
    # get the sos parameters for sos eq (i.e. more than 2 sos).
    assert 1 < len(f0) == len(dBgain) == len(Q) == len(eq_type)
    nsos = len(f0)
    sos = []
    for i in range(nsos):
        f0_, dBgain_, Q_, eq_type_ = f0[i], dBgain[i], Q[i], eq_type[i]
        sos_ = get_eq_sos(sr, f0_, dBgain_, Q_, eq_type_)
        sos.append(sos_)
    return np.array(sos)

def get_eqsos_fr(sr, f0: list, dBgain: list, Q: list, eq_type: list):
    sos = get_eqsos_sos(sr, f0, dBgain, Q, eq_type)
    f, z = signal.sosfreqz(sos, fs=sr)
    return f, 20*np.log10(abs(z)), np.unwrap(np.angle(z)) # f, amp, phase

def get_eqsos(y, sr, f0: list, dBgain: list, Q: list, eq_type: list, axis=0):
    sos = get_eqsos_sos(sr, f0, dBgain, Q, eq_type)
    return signal.sosfilt(sos, y, axis=axis)
        
# Time-frequency transform
def get_fftm(y, axis=-1):
    # This returns frequency magnitudes (real positive numbers) instead of complex numbers.
    return np.abs(fft.fft(y, axis=axis, norm='backward'))

def get_rfftm(y, axis=-1):
    # This returns frequency magnitudes (real positive numbers) instead of complex numbers.
    return np.abs(fft.rfft(y, axis=axis, norm='backward'))

def get_rfftmf(y, sr, axis=-1):
    # This returns frequency magnitudes and the frequencies consistent with sr.
    y_rfftm = np.abs(fft.rfft(y, axis=axis, norm='backward'))
    y_rfftf = np.arange(y_rfftm.size)*sr/y.size
    return y_rfftm, y_rfftf

def get_fft(y, axis=-1):
    return fft.fft(y, axis=axis, norm='backward')

def get_ifft(y_fft, axis=-1):
    return fft.ifft(y_fft, axis=axis, norm='backward')

def get_rfft(y, axis=-1):
    return fft.rfft(y, axis=axis, norm='backward')

def get_irfft(y_rfft, axis=-1):
    return fft.irfft(y_rfft, axis=axis, norm='backward')

def get_dct(y, axis=-1, dct_type=2):
    return fft.dct(au, dct_type, axis=axis, norm='backward')

def get_idct(y_dct, axis=-1, dct_type=2):
    return fft.idct(y_dct, dct_type, axis=axis, norm='backward')

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

class stft_class():

    def __init__(self, sr, T=0.025, overlap=0.75, fft_ratio=1.0, win='blackmanharris', fft_type='m, p', GLA_n_iter=100, GLA_random_phase_type='mono'):
        """
        Parameters:
        sr: int (Hz). Sample rate, ususally 44100 or 48000.
        T: float (seconds). Time length of a each window. For 48000kHz, T=0.01067 means n=512.
        overlap: float (ratio between 0 and 1). Overlap ratio between each two adjacent windows.
        fft_ratio: float (ratio >= 1). The fft ratio relative to T.
        win: str. Please refer to scipy's window functions. Window functions like kaiser will require a tuple input including additional parameters. e.g. ('kaiser', 14.0)
        fft_type: str ('m', 'm, p', 'z' or 'zr, zi'). Please refer to the illustration of the returns of self.forward(). If fft_type=='m', istft will use the Griffin-Lim algorithm (GLA).
        GLA_n_iter: int. The iteration times for GLA.
        GLA_random_phase_type: str ('mono' or 'stereo'). Whether the starting random phases for GLA are different between 2 stereo channels.
        """
        self.sr, self.nperseg, self.noverlap, self.nfft = sr, int(sr*T), int(sr*T*overlap), int(sr*T*fft_ratio)
        self.nhop = self.nperseg - self.noverlap
        self.win, self.fft_type = signal.windows.get_window(win, self.nperseg, fftbins=True), fft_type
        self.GLA_n_iter, self.GLA_random_phase_type = GLA_n_iter, GLA_random_phase_type

    def fw(self, au):
        """
        Short-Time Fourier Transform

        Parameters:
        au: ndarray (dtype = float between -1 and 1). Need to have 1 or 2 dimensions like normal single-channel or multi-channel audio. 

        Returns:
        f: 1d array. As scipy.signal.stft returns.
        t: 1d array. As scipy.signal.stft returns.
        m: if self.fft_type='m'. The magnitudes array of shape (f.size, t.size) or (f.size, t.size, au.shape[-1]). PLEASE NOTE that the istft will use phases of a white noise!
        m, p: if self.fft_type='m, p'. The magnitudes array and phases array of shapes (f.size, t.size) or (f.size, t.size, au.shape[-1]). The phase range is [-pi, pi].
        z: if self.fft_type='z'. The complex array of shape (f.size, t.size) or (f.size, t.size, au.shape[-1]).
        zr, zi: if self.fft_type='zr, zi'. The complex array' real array and imaginary array of shapes (f.size, t.size) or (f.size, t.size, au.shape[-1]).
        """
        f, t, z = signal.stft(au, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, axis=0)
        z = z.swapaxes(1, -1)
        print(f'au.shape = {au.shape}')
        print(f'f.shape = {f.shape}')
        print(f't.shape = {t.shape}')
        print(f'z.shape = {z.shape}')
        if self.fft_type == 'm':
            m = np.abs(z)
            print(f'm.shape = {m.shape}')
            return f, t, m
        elif self.fft_type == 'm, p':
            m, p = np.abs(z), np.unwrap(np.angle(z))
            print(f'm.shape = {m.shape}')
            print(f'p.shape = {p.shape}')
            return f, t, m, p
        elif self.fft_type == 'z':
            return f, t, z
        elif self.fft_type == 'zr, zi':
            return f, t, z.real, z.imag
        else:
            raise ValueError('Parameter self.fft_type has to be "m", "m, p", "z" or "zr, zi".')

    def bw(self, m=None, p=None, z=None, zr=None, zi=None, nsample=None):
        """
        Inverse Short-Time Fourier Transform

        Parameters:
        in_tup: an ndarray or a tuple containing 2 ndarrays corresponding to self.fft_type. Please refer to the illustration of the returns of self.forward().
        
        Returns:
        au_re: ndarray. Audio array after inverse short-time fourier transform.
        """
        if self.fft_type == 'm, p':
            assert m is not None, f'm is None'
            assert p is not None, f'p is None'
            z = np.empty(m.shape, dtype=np.complex128)
            z.real, z.imag = m*np.cos(p), m*np.sin(p)
        elif self.fft_type == 'm':
            assert m is not None, f'm is None'
            assert nsample is not None, f'nsample is None'
            p = self.get_random_phase(nsample, m.ndim)
            z = np.empty(m.shape, dtype=np.complex128)
            z.real, z.imag = m*np.cos(p), m*np.sin(p)
            for i in range(0, self.GLA_n_iter):
                t, au_re = signal.istft(z, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, time_axis=1, freq_axis=0)
                f, t, z = signal.stft(au_re, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, axis=0)
                z = z.swapaxes(1, -1)
                p = np.angle(z)
                z.real, z.imag = m*np.cos(p), m*np.sin(p)   
        elif self.fft_type == 'z':
            assert z is not None, f'z is None'
        elif self.fft_type == 'zr, zi':
            assert zr is not None, f'zr is None'
            assert zi is not None, f'zi is None'
            z = np.empty(in_tup[0].shape, dtype=np.complex128)
            z.real, z.imag = zr, zi
        else:
            raise ValueError('Parameter self.fft_type has to be "m", "m, p", "z" or "zr, zi".')
        t, au_re = signal.istft(z, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, time_axis=1, freq_axis=0)
        print(f'au_re.shape = {au_re.shape}')
        return au_re

    def re(self, au):
        """
        Reconstruct an audio array using stft and then istft. Please refer to the illustration of the returns of self.forward().

        Parameters:
        au: ndarray (dtype = float between -1 and 1). Need to have 1 or 2 dimensions like normal single-channel or multi-channel audio. 
        """          
        if self.fft_type == 'm, p':
            f, t, m, p = self.fw(au)
            au_re = self.bw(m=m, p=p)
        elif self.fft_type == 'm':
            # Using the Griffin-Lim algorithm.
            f, t, m = self.fw(au)
            nsample = au.shape[0]
            print(f'nsample = {nsample}')
            au_re = self.bw(m=m, nsample=nsample)
        elif self.fft_type == 'z':
            f, t, z = self.fw(au)
            au_re = self.bw(z=z)
        elif self.fft_type == 'zr, zi':
            f, t, zr, zi = self.fw(au)
            au_re = self.bw(zr=zr, zi=zi)
        else:
            raise ValueError('Parameter self.fft_type has to be "m", "m, p", "z" or "zr, zi".')
        return au_re        

    def get_random_phase(self, nsample, m_ndim):
        if m_ndim == 3:
            if self.GLA_random_phase_type == 'mono':
                noise = 0.5*np.random.uniform(-1, 1, nsample)
                noise = np.stack((noise, noise), axis=-1)
            elif self.GLA_random_phase_type == 'stereo':
                noise = 0.5*np.random.uniform(-1, 1, (nsample, 2))
            else:
                raise ValueError('self.GLA_random_phase_type != "mono" or "stereo"')
        elif m_ndim == 2:
            noise = 0.5*np.random.uniform(-1, 1, nsample)
        else:
            raise ValueError('m_ndim != 2 or 3')
        f_noise, t_noise, z_noise = signal.stft(noise, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, axis=0)
        z_noise = z_noise.swapaxes(1, -1)
        p_noise = np.angle(z_noise*np.exp(0.5*np.pi*1.0j))
        print(f'p_noise.shape = {p_noise.shape}')
        return p_noise    

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
def get_idx_array(y):
    return np.arange(y.size)

def sample2time(n, sr):
    # start from 0
    return n/sr

def time2sample(t, sr):
    # start from 0
    return (sr*t).astype(np.int64)

def get_pitch_given(au, sr, du=None, given_freq=440, given_cent=100, cent_step=1):
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
        au = np.average(au, axis=-1)
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
