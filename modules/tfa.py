"""
Audio Time-Frequency Analysis
"""
import numpy as np
from scipy import fft, signal

# IIR filter
# Input: signal and filter coefficients/parameters
# Output: filtered signal
# Funtions with "sos" uses second-order sections for numerical stability.
# Funtions with "2" uses fordward-backward filtering to realize zero phase.

def iir(y, b, a, axis=0):
    return signal.lfilter(b, a, y, axis=axis)

def iir2(y, b, a, axis=0):
    return signal.filtfilt(b, a, y, axis=axis)

def iirsos(y, sos, axis=0):
    return signal.sosfilt(sos, y, axis=axis)

def iirsos2(y, sos, axis=0):
    return signal.sosfiltfilt(sos, y, axis=axis)

def bq(y, sr, bqfunc, freq, Q, gain=None, axis=0):
    """
    Single biquad filter

    Parameters:
    y: ndarray. The signal to be filtered.
    sr: positive int (Hz). Sample rate of y.
    bqfunc: function. Which function to use to get the sos coefficients of biquad filter,
        e.g. bq_lowpass. Please refer to the "IIR filter coefficient" section.
    freq: positive float. The "significant frequency".
    Q: positive float. Quality factor.
    gain: float (dB). For peak, low shelf and high shelf filters only.
    axis: int. Which axis of y to filter along.

    Returns:
    y_filtered: ndarray. Filtered signal.
    """
    sos = bqfunc(sr, freq, Q, gain)
    return iirsos(y, sos, axis=axis)

def bq2(y, sr, bqfunc, freq, Q, gain=None, axis=0):
    sos = bqfunc(sr, freq, Q, gain)
    return iirsos2(y, sos, axis=axis)

def bqsos(y, sr, bqfunc_list, freq_list, Q_list, gain_list=None, axis=0):
    # Cascaded-sos biquad filter, with list inputs.
    sos = get_sos_bq(sr, bqfunc_list, freq_list, Q_list, gain_list)
    return iirsos(y, sos, axis=axis)

def bqsos2(y, sr, bqfunc_list, freq_list, Q_list, gain_list=None, axis=0):
    sos = get_sos_bq(sr, bqfunc_list, freq_list, Q_list, gain_list)
    return iirsos2(y, sos, axis=axis)

def butter(y, sr, btype, order, freq, axis=0):
    # N (lp or hp) or 2*N (bp or bs) -order Butterworth filter
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    b, a = get_ba_butter(sr, btype, order, freq)
    return iir(y, b, a, axis=axis)

def butter2(y, sr, btype, order, freq, axis=0):
    b, a = get_ba_butter(sr, btype, order, freq)
    return iir2(y, b, a, axis=axis)

def buttersos(y, sr, btype, order, freq, axis=0):
    sos = get_sos_butter(sr, btype, order, freq)
    return iirsos(y, sos, axis=axis)

def buttersos2(y, sr, btype, order, freq, axis=0):
    sos = get_sos_butter(sr, btype, order, freq)
    return iirsos2(y, sos, axis=axis)

# IIR filter frequency response
# Input: filter coefficients/parameters
# Output: frequency response (frequency, amplitude and phase arrays)
# In case of zero division warning: (old_settings =) np.seterr(divide='ignore')

def fr_iir(sr, b, a):
    f, z = signal.freqz(b, a, fs=sr)
    return f, 20*np.log10(abs(z)), np.unwrap(np.angle(z))

def fr_iirsos(sr, sos):
    f, z = signal.sosfreqz(sos, fs=sr)
    return f, 20*np.log10(abs(z)), np.unwrap(np.angle(z))

def fr_bq(sr, bqfunc, freq, Q, gain=None):
    sos = bqfunc(sr, freq, Q, gain)
    return fr_irrsos(sr, sos)

def fr_bqsos(sr, bqfunc_list, freq_list, Q_list, gain_list=None):
    sos = get_sos_bq(sr, bqfunc_list, freq_list, Q_list, gain_list)
    return fr_irrsos(sr, sos)

def fr_butter(sr, btype, order, freq, axis=0):
    b, a = get_ba_butter(sr, btype, order, freq)
    return fr_irr(sr, b, a)

def fr_buttersps(sr, btype, order, freq, axis=0):
    sos = get_sos_butter(sr, btype, order, freq)
    return fr_irrsos(sr, sos)

# IIR filter coefficient
# Input: filter parameters
# Output: filter coefficients)
# Reference: Audio EQ Cookbook (W3C Working Group Note, 08 June 2021)

def cascade_sos(sos_list):
    # input sos list (or tuple): [sos1, sos2,..., sosn]
    return np.concatenate(sos_list, axis=0)

def repeat_sos(sos, n):
    # Broadcast a sos from shape(1, 6) to shape(n, 6).
    return np.broadcast_to(sos, (n, 6))

def get_ba_butter(sr, btype, order, freq):
    # Get the b and a parameters of a butterworth IIR filter.
    # signal.butter
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    return signal.butter(order, freq, btype=btype, fs=sr)

def get_sos_butter(sr, btype, order, freq):
    # Get the sos of a butterworth IIR filter.
    # signal.butter
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    return signal.butter(order, freq, btype=btype, output='sos', fs=sr)

def get_sos_iir(sr, ftype, btype, order, freq, rp=None, rs=None):
    # Get the sos of certain types of IRR filter.
    # signal.iirfilter
    # ftype: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    return signal.iirfilter(order, freq, rp=rp, rs=rs, btype=btype, \
                            ftype=ftype, output='sos', fs=sr)

def get_sos_bq(sr, bqfunc_list, freq_list, Q_list, gain_list=None):
    # Get the sos coefficient array of cascaded biquad filters.
    # Returned sos array shape is (number of sos, 6).
    # The length of input lists (should be the same) is the number of sos.
    nsos = len(freq_list)
    if gain_list is None:
        gain_list = [None]*nsos
    sos = np.empty((0, 6))
    for i in range(nsos):
        bqfunc = bqfunc_list[i]
        freq = freq_list[i]
        Q = Q_list[i]
        gain = gain_list[i]
        sos_ = bqfunc(sr, freq, Q, gain)
        sos = np.append(sos, sos_, axis=0)
    return sos

def bq_allpass(sr, freq, Q, gain=None):
    # biquad all pass filter
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    norm = 1+alpha    
    b = np.array([1-alpha, -2*cosw, 1+alpha])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])
        
def bq_lowpass(sr, freq, Q, gain=None):
    # biquad low pass filter
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    norm = 1+alpha    
    b = np.array([0.5*(1-cosw), 1-cosw, 0.5*(1-cosw)])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])

def bq_highpass(sr, freq, Q, gain=None):
    # biquad high pass filter
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    norm = 1+alpha    
    b = np.array([0.5*(1+cosw), -1-cosw, 0.5*(1+cosw)])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])

def bq_bandpass(sr, freq, Q, gain=None):
    # biquad band pass filter with constant 0 dB peak gain.
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    norm = 1+alpha    
    b = np.array([alpha, 0.0, -alpha])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])

def bq_bandpass2(sr, freq, Q, gain=None):
    # biquad band pass filter with constant Q dB peak gain.
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    norm = 1+alpha    
    b = np.array([Q*alpha, 0.0, -Q*alpha])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])

def bq_bandstop(sr, freq, Q, gain=None):
    # biquad band stop (notch) filter
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    norm = 1+alpha    
    b = np.array([1.0, -2*cosw, 1.0])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])
    
def bq_peak(sr, freq, Q, gain):
    # biquad peaking (bell) EQ
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    A = np.power(10, gain/40)
    norm = 1+alpha/A    
    b = np.array([1+alpha*A, -2*cosw, 1-alpha*A])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha/A)/norm])
    return np.array([np.append(b, a)])

def bq_lowshelf(sr, freq, Q, gain):
    # biquad low shelf
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    A = np.power(10, gain/40)
    norm = (A+1) + (A-1)*cosw + 2*np.sqrt(A)*alpha    
    b0 = A*((A+1) - (A-1)*cosw + 2*np.sqrt(A)*alpha)
    b1 = 2*A*((A-1) - (A+1)*cosw)
    b2 = A*((A+1) - (A-1)*cosw - 2*np.sqrt(A)*alpha)
    a1 = -2*((A-1) + (A+1)*cosw)
    a2 = (A+1) + (A-1)*cosw - 2*np.sqrt(A)*alpha
    b = A*np.array([b0, b1, b2])/norm
    a = A*np.array([1.0, a1/norm, a2/norm])
    return np.array([np.append(b, a)])

def bq_highshelf(sr, freq, Q, gain):
    # biquad high shelf
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    A = np.power(10, gain/40)
    norm = (A+1) - (A-1)*cosw + 2*np.sqrt(A)*alpha    
    b0 = A*((A+1) + (A-1)*cosw + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*cosw)
    b2 = A*((A+1) + (A-1)*cosw - 2*np.sqrt(A)*alpha)
    a1 = 2*((A-1) - (A+1)*cosw)
    a2 = (A+1) - (A-1)*cosw - 2*np.sqrt(A)*alpha
    b = A*np.array([b0, b1, b2])/norm
    a = A*np.array([1.0, a1/norm, a2/norm])
    return np.array([np.append(b, a)])

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
    y_rfftf = fft.fftfreq(y.size, d=1/sr)
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
    # STFT and ISTFT using python's class.
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
        f: 1d array. As signal.stft returns.
        t: 1d array. As signal.stft returns.
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
def get_pitch_given(au, sr, du=None, given_freq=440, given_cent=100, cent_step=1):
    """
    Detect the pitch of audio (specifically piano single note) given a pitch, cent band and cent step, using discrete time fourier transform in limited frequency range.
    The computation will be quite slow since it does not use FFT, but it's much more accurate than signal.stft in terms of frequency resolution. 
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
    F = given_freq*np.exp2(np.arange(-given_cent, given_cent+1, cent_step)/1200)
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
    win: str or tuple. The window to apply to every frame. No need to provide window size. Please refer to signal.get_windows.

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
