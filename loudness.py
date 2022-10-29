"""
The LUFS calculations here are all based on:
ITU documentation: https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf
EBU documentation: https://tech.ebu.ch/docs/tech/tech3341.pdf
pyloudnorm by csteinmetz1: https://github.com/csteinmetz1/pyloudnorm
loudness.py by BrechtDeMan: https://github.com/BrechtDeMan/loudness.py

I just rewrote some codes and added a momentary loudness calculation for more convinent batch processing of audio files.
"""
import numpy as np
from scipy import signal

def amp2db(amp: float): # zero or positive amp value range between 0 and 1.
    return 20*np.log10(amp)

def db2amp(db: float): # zero or negative db value.
    return np.power(10, db/20)

def get_sinewave(f, phase=0, A=1, du=1, sr=48000, stereo=True):
    """
    Generate a pure sine wave for loudness testing.
    f: float. Frequency.
    phase: float. Initial phase.
    A: float. Maxinum amplitude.
    """
    t = np.arange(0, int(du*sr))/sr
    size = t.size
    y = A*np.sin(2*np.pi*f*t + phase)
    if stereo:
        y = y.reshape(size, 1)
        return np.broadcast_to(y, (size, 2))
    else:
        return y

def get_prefilter_coff(sr, G, Q, fc, filter_type, passband_gain=1.0):
    """ 
    Parameters
    sr: float. Sampling rate in Hz.
    G : float. Gain of the filter in dB.
    Q : float. Q of the filter.
    fc : float. Center frequency of the shelf in Hz.
    filter_type: str. Shape of the filter.
    """    
    A  = np.power(10, G/40.0)
    w0 = 2.0*np.pi*(fc/sr)
    alpha = np.sin(w0)/(2.0*Q)
    if filter_type == 'high_shelf':
        b0 =      A * ( (A+1) + (A-1) * np.cos(w0) + 2 * np.sqrt(A) * alpha )
        b1 = -2 * A * ( (A-1) + (A+1) * np.cos(w0)                          )
        b2 =      A * ( (A+1) + (A-1) * np.cos(w0) - 2 * np.sqrt(A) * alpha )
        a0 =            (A+1) - (A-1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 =      2 * ( (A-1) - (A+1) * np.cos(w0)                          )
        a2 =            (A+1) - (A-1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    elif filter_type == 'high_pass':
        b0 =  (1 + np.cos(w0))/2
        b1 = -(1 + np.cos(w0))
        b2 =  (1 + np.cos(w0))/2
        a0 =   1 + alpha
        a1 =  -2 * np.cos(w0)
        a2 =   1 - alpha
    elif filter_type == 'peaking':
        b0 =   1 + alpha * A
        b1 =  -2 * np.cos(w0)
        b2 =   1 - alpha * A
        a0 =   1 + alpha / A
        a1 =  -2 * np.cos(w0)
        a2 =   1 - alpha / A
    return np.array([b0, b1, b2])/a0, np.array([a0, a1, a2])/a0

def get_prefilter_coff_Kw0(sr=48000):
    if sr == 48000:
        a0 = 1
        a1 = -1.69065929318241
        a2 = 0.73248077421585
        b0 = 1.53512485958697
        b1 = -2.69169618940638
        b2 = 1.19839281085285
        d0, c0 = np.array([b0,b1,b2]), np.array([a0,a1,a2])

        a0 = 1
        a1 = -1.99004745483398
        a2 = 0.99007225036621
        b0 = 1.0
        b1 = -2.0
        b2 = 1.0
        d1, c1 = np.array([b0,b1,b2]), np.array([a0,a1,a2])
        return d0, c0, d1, c1
    else:
        raise ValueError(f'Sample rate {sr} is not supported.')

def get_prefilter_coff_Kw1(sr):
    d0, c0 = get_prefilter_coff(sr, 4.0, 1/np.sqrt(2), 1500.0, 'high_shelf')
    d1, c1 = get_prefilter_coff(sr, 0.0, 0.5, 38.0, 'high_pass')
    return d0, c0, d1, c1

def get_prefilter_coff_Kw2(sr):
    # This is closer to the ITU documentation than get_prefilter_coff_Kw1(sr).
    # pre-filter 1
    f0 = 1681.9744509555319
    G  = 3.99984385397
    Q  = 0.7071752369554193
    K  = np.tan(np.pi * f0 / sr) 
    Vh = np.power(10.0, G / 20.0)
    Vb = np.power(Vh, 0.499666774155)
    a0_ = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0_
    b1 = 2.0 * (K * K -  Vh) / a0_
    b2 = (Vh - Vb * K / Q + K * K) / a0_
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / a0_
    a2 = (1.0 - K / Q + K * K) / a0_
    d0, c0 = np.array([b0,b1,b2]), np.array([a0,a1,a2])
    # pre-filter 2
    f0 = 38.13547087613982
    Q  = 0.5003270373253953
    K  = np.tan(np.pi * f0 / sr)
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
    a2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
    b0 = 1.0
    b1 = -2.0
    b2 = 1.0
    d1, c1 = np.array([b0,b1,b2]), np.array([a0,a1,a2])
    return d0, c0, d1, c1

def get_prefilter_coff_Fenton(sr):
    d0, c0 = get_prefilter_coff(sr, 5.0, 1/np.sqrt(2), 1500.0, 'high_shelf')
    d1, c1 = get_prefilter_coff(sr, 0.0, 0.5, 130.0, 'high_pass')
    d2, c2 = get_prefilter_coff(sr, 0.0, 1/np.sqrt(2), 500.0, 'high_pass')
    return d0, c0, d1, c1, d2, c2

def prefilter_Kw(au, sr, d0, c0, d1, c1):
    au = signal.lfilter(d0, c0, au, axis=0)
    au = signal.lfilter(d1, c1, au, axis=0)
    return au   

def prefilter_Fenton(au, sr, d0, c0, d1, c1, d2, c2):
    au = signal.lfilter(d0, c0, au, axis=0)
    au = signal.lfilter(d1, c1, au, axis=0)
    au = signal.lfilter(d2, c2, au, axis=0)
    return au    

def get_lufs_mono(au, sr):
    """
    The input audio should have been prefiltered.
    """
    lufs = -0.691 + 10*np.log10(np.average(np.square(au)))
    return lufs

def get_lufs_multi(au, sr):
    """
    The input audio should have been prefiltered.
    """
    lufs = -0.691 + 10*np.log10(np.sum(np.average(np.square(au), axis=0)))
    return lufs

def get_Mlufs(au, sr, win=0.4, overlap=0.75):
    """
    win: float, seconds. Time length of each window.
    overlap: float, proportion. Proportion of overlapping between windows. 
    """
    step = int(sr*win)
    hop = int(step*(1-overlap))
    Mlufs = np.empty(0)
    d0, c0, d1, c1 = get_prefilter_coff_Kw2(sr)
    if au.ndim == 2:
        q1, q2 = divmod(au.shape[0], hop)
        q3 = step - hop - q2
        if q3 > 0:
            au = np.append(au, np.zeros((q3, au.shape[-1])), axis=0)
        elif q3 < 0:
            raise ValueError('q3 < 0')
        for i in range(0, q1):
            au_f = au[i*hop: i*hop+step, :]
            au_f = prefilter_Kw(au_f, sr, d0, c0, d1, c1)
            lufs = get_lufs_multi(au_f, sr)
            Mlufs = np.append(Mlufs, lufs)
    elif au.ndim == 1:
        q1, q2 = divmod(au.shape[0], hop)
        q3 = step - hop - q2
        if q3 > 0:
            au = np.append(au, np.zeros(q3), axis=0)
        elif q3 < 0:
            raise ValueError('q3 < 0')
        for i in range(0, q1):
            au_f = au[i*hop: i*hop+step]
            au_f = prefilter_Kw(au_f, sr, d0, c0, d1, c1)
            lufs = get_lufs_mono(au_f, sr)
            Mlufs = np.append(Mlufs, lufs)
    else:
        raise ValueError(f'au.ndim = {au.ndim} is not supported.')
    return np.amax(Mlufs)

def get_Ilufs(au, sr, win=0.4, overlap=0.75):
    """
    win: float, seconds. Time length of each window.
    overlap: float, proportion. Proportion of overlapping between windows. 
    """
    step = int(sr*win)
    hop = int(step*(1-overlap))
    Lk, Z = np.empty(0), np.empty(0)
    d0, c0, d1, c1 = get_prefilter_coff_Kw2(sr)
    if au.ndim == 2:
        nchannel = au.shape[-1]
        q1, q2 = divmod(au.shape[0], hop)
        q3 = step - hop - q2
        if q3 > 0:
            au = np.append(au, np.zeros((q3, nchannel)), axis=0)
        elif q3 < 0:
            raise ValueError('q3 < 0')       
        for i in range(0, q1):
            au_f = au[i*hop: i*hop+step, :]
            au_f = prefilter_Kw(au_f, sr, d0, c0, d1, c1)
            z = np.sum(np.average(np.square(au_f), axis=0))
            lk = -0.691 + 10*np.log10(z)
            Lk, Z = np.append(Lk, lk), np.append(Z, z)
    elif au.ndim == 1:
        q1, q2 = divmod(au.shape[0], hop)
        q3 = step - hop - q2
        if q3 > 0:
            au = np.append(au, np.zeros(q3), axis=0)
        elif q3 < 0:
            raise ValueError('q3 < 0')
        for i in range(0, q1):
            au_f = au[i*hop: i*hop+step]
            au_f = prefilter_Kw(au_f, sr, d0, c0, d1, c1)
            z = np.average(np.square(au_f), axis=0)
            lk = -0.691 + 10*np.log10(z)
            Lk, Z = np.append(Lk, lk), np.append(Z, z)
    else:
        raise ValueError(f'au.ndim = {au.ndim} is not supported.')
    Z0 = Z[Lk > -70]
    td = -0.691 + 10*np.log10(np.average(Z0)) - 10
    Z = Z[Lk > td]
    Ilufs = -0.691 + 10*np.log10(np.average(Z))
    return Ilufs

def norm_Mlufs(au, sr, target=-18.0):
    au *= db2amp(target - get_Mlufs(au, sr))
    return au

def check_clipping(au):
    if np.amax(np.abs(au)) >= 1:
        raise ValueError('Clipping has occurred.')

def print_peak(au):
    au_abs = np.abs(au)
    peak_amp = np.amax(au_abs)
    peak_db = amp2db(peak_amp)
    print(f'peak_amp = {round(peak_amp, 5)}, peak_db = {round(peak_db, 2)}')

def get_peak(au):
    return np.amax(np.abs(au))

def get_peak_LR(au):
    peak_LR = np.amax(np.abs(au), axis=0)
    return peak_LR[0], peak_LR[1]

def norm_mid_peak(au, amp=0.35):
    """
    normalize the peak amplitude of the mid channel of a stereo wav file under the normal -3db pan law.
    input array, output array, no read or write audio files.
    """
    au *= amp/np.amax(np.abs(np.average(au, axis=-1)))
    return au

def norm_mono_peak(au, amp=0.5):
    """
    normalize the peak amplitude of a mono audio.
    input array, output array, no read or write audio files.
    """
    au *= amp/np.amax(np.abs(au))
    return au

def change_LR_ratio(au, scale):
    """
    scale: zero or positive float. 1 means unchanged. >1 means to intensify the LR peaks' difference and <1 means to attenuate the LR peaks' difference.
    If peak_L >= peak_R, before changing, peak_L=peak_R*ratio. After changing, peak_L=peak_R*(1+scale*(ratio-1)).
    """
    peak_L, peak_R = get_peak_LR(au)
    if peak_L >= peak_R:
        ratio = peak_L/peak_R
        au *= np.array([(1+(ratio-1)*scale)/ratio, 1])
    else:
        ratio = peak_R/peak_L
        au *= np.array([1, (1+(ratio-1)*scale)/ratio])
    return au
