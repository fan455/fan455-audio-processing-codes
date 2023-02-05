# Filter

import numpy as np
from scipy import signal

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
# Output: frequency response i.e. frequency (Hz), magnitude (dB) and phase (rad) arrays.
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
    b = np.array([b0, b1, b2])/norm
    a = np.array([1.0, a1/norm, a2/norm])
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
    b = np.array([b0, b1, b2])/norm
    a = np.array([1.0, a1/norm, a2/norm])
    return np.array([np.append(b, a)])
