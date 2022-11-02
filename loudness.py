# Audio loudness calculation and normalization.
# The LUFS calculations here are all based on:
# ITU documentation: https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf
# EBU documentation: https://tech.ebu.ch/docs/tech/tech3341.pdf
# pyloudnorm by csteinmetz1: https://github.com/csteinmetz1/pyloudnorm
# loudness.py by BrechtDeMan: https://github.com/BrechtDeMan/loudness.py
# Special thanks to these authors!
# I just rewrote some codes to enable short-term and momentary loudness calculations and normalizations for more convinent batch processing of audio files.
# True peak algorithm has not been implemented here.
# To get the integrated lufs, use the 'Ilufs_meter().get()' function. To get the short-term or momentary lufs, use the 'Mlufs_meter().get()' function.
# To normalize the integrated lufs, use the 'norm_Ilufs()' function. To normalize the short-term or momentary lufs, use the 'norm_Mlufs()' function.
# Both mono and stereo input audio arrays with amplitudes between -1 and 1 are supported.

import numpy as np
from scipy import signal

def amp2db(amp: float): # zero or positive amp value range between 0 and 1.
    return 20*np.log10(amp)

def db2amp(db: float): # zero or negative db value.
    return np.power(10, db/20)

def check_clipping(au):
    if np.amax(np.abs(au)) >= 1:
        raise ValueError('Clipping has occurred.')

def get_sinewave(f, phase=0, A=1, du=1, sr=48000, stereo=True, ls=None, ts=None):
    """
    Generate a pure sine wave for loudness testing.
    f: float. Frequency.
    phase: float. Initial phase.
    A: float. Maxinum amplitude.
    du: float, seconds. Duration of sinewave.
    ls: float, seconds. Duration of leading silence.
    ts: float, seconds. Duration of trailing silence.
    """
    t = np.arange(0, int(du*sr))/sr
    size = t.size
    y = A*np.sin(2*np.pi*f*t + phase)
    if ls:
        y = np.append(np.zeros(int(ls*sr)), y)
    if ts:
        y = np.append(y, np.zeros(int(ts*sr)))
    if stereo:
        y = y.reshape(size, 1)
        return np.broadcast_to(y, (size, 2))
    else:
        return y
    
def norm_lufs(au, sr, old, new=-18.0):
    return au*db2amp(new - old)

def norm_Mlufs(au, sr, old, new=-18.0):
    return au*db2amp(new - old)

def norm_Ilufs(au, sr, old, new=-23.0):
    return au*db2amp(new - old)

class Mlufs_meter():
    # This allows the pre-computation of prefilter coefficients for better performance.
    def __init__(self, sr):
        self.sr = sr
        if sr == 48000:
            self.d0 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
            self.c0 = np.array([1.0 , -1.69065929318241, 0.73248077421585])
            self.d1 = np.array([1.0, -2.0, 1.0])
            self.c1 = np.array([1.0, -1.99004745483398, 0.99007225036621])
        elif sr == 44100:
            self.d0 = np.array([1.5308412300498355, -2.6509799951536985, 1.1690790799210682])
            self.c0 = np.array([1.0, -1.6636551132560204, 0.7125954280732254])
            self.d1 = np.array([1.0, -2.0, 1.0])
            self.c1 = np.array([1.0, -1.9891696736297957, 0.9891990357870394])
        else:
            # coefficients calculation by BrechtDeMan. This is closer to the ITU documentation than get_prefilter_coeff_Kw1(sr).
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
            self.d0, self.c0 = np.array([b0, b1, b2]), np.array([a0, a1, a2])
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
            self.d1, self.c1 = np.array([b0, b1, b2]), np.array([a0, a1, a2])
            
    def prefilter(self, au):
        au = signal.lfilter(self.d0, self.c0, au, axis=0)
        au = signal.lfilter(self.d1, self.c1, au, axis=0)
        return au

    def get(self, au, T=0.4, overlap=0.75, cut_start=None):
        """
        Get the maxinum momentary lufs of a mono or stereo audio input. The audio array will be padded zeros at the end to complete the last window.
        T: float, seconds. Time length of each window. You can change it to 3 to get the short-term lufs (Slufs).
        overlap: float, proportion. Proportion of overlapping between windows.
        cut_start: float, seconds. The start seconds to only analyze.
        Only works for mono or stereo audio because I just summed all the channels and didn't calculate the different weights in case of a 5-channel audio input.
        You can modify the return 'np.amax(Mlufs)' to 'Mlufs' so you can get the full Mlufs array corresponding to the windowed audio.
        """
        step, hop = int(self.sr*T), int(self.sr*T*(1-overlap))
        Mlufs = np.empty(0)
        if au.ndim == 2:
            if cut_start:
                au = au[0: int(sr*cut_start), :]
            q1, q2 = divmod(au.shape[0], hop)
            q3 = step - hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros((q3, au.shape[-1])), axis=0)
            for i in range(0, q1):
                au_f = au[i*hop: i*hop+step, :]
                au_f = self.prefilter(au_f)
                mlufs = -0.691 + 10*np.log10(np.sum(np.average(np.square(au_f), axis=0)))
                Mlufs = np.append(Mlufs, mlufs)
        elif au.ndim == 1:
            if cut_start:
                au = au[0: int(sr*cut_start)]
            q1, q2 = divmod(au.shape[0], hop)
            q3 = step - hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros(q3), axis=0)
            for i in range(0, q1):
                au_f = au[i*hop: i*hop+step]
                au_f = self.prefilter(au_f)
                mlufs = -0.691 + 10*np.log10(np.average(np.square(au_f)))
                Mlufs = np.append(Mlufs, mlufs)
        else:
            raise ValueError(f'au.ndim = {au.ndim} is not supported.')
        return np.amax(Mlufs)
    
class Ilufs_meter():
    # This allows the pre-computation of prefilter coefficients for better performance.
    def __init__(self, sr):
        self.sr = sr
        if sr == 48000:
            self.d0 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
            self.c0 = np.array([1.0 , -1.69065929318241, 0.73248077421585])
            self.d1 = np.array([1.0, -2.0, 1.0])
            self.c1 = np.array([1.0, -1.99004745483398, 0.99007225036621])
        elif sr == 44100:
            self.d0 = np.array([1.5308412300498355, -2.6509799951536985, 1.1690790799210682])
            self.c0 = np.array([1.0, -1.6636551132560204, 0.7125954280732254])
            self.d1 = np.array([1.0, -2.0, 1.0])
            self.c1 = np.array([1.0, -1.9891696736297957, 0.9891990357870394])
        else:
            # coefficients calculation by BrechtDeMan. This is closer to the ITU documentation than get_prefilter_coeff_Kw1(sr).
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
            self.d0, self.c0 = np.array([b0,b1,b2]), np.array([a0,a1,a2])
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
            self.d1, self.c1 = np.array([b0,b1,b2]), np.array([a0,a1,a2])
            
    def prefilter(self, au):
        au = signal.lfilter(self.d0, self.c0, au, axis=0)
        au = signal.lfilter(self.d1, self.c1, au, axis=0)
        return au

    def get(self, au, T=0.4, overlap=0.75, cut_start=None):
        """
        Get the integrated lufs of a mono or stereo audio input. The audio array will be padded zeros at the end to complete the last window.
        T: float, seconds. Time length of each window. It means the same as 'gating' in the ITU doc.
        overlap: float, proportion. Proportion of overlapping between windows.
        Only works for mono or stereo audio because I just summed all the channels and didn't calculate the different weights in case of a 5-channel audio input.
        """
        step, hop = int(self.sr*T), int(self.sr*T*(1-overlap))
        Lk, Z = np.empty(0), np.empty(0)
        if au.ndim == 2:
            if cut_start:
                au = au[0: int(sr*cut_start), :]
            nchannel = au.shape[-1]
            q1, q2 = divmod(au.shape[0], hop)
            q3 = step - hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros((q3, nchannel)), axis=0)       
            for i in range(0, q1):
                au_f = au[i*hop: i*hop+step, :]
                au_f = self.prefilter(au_f)
                z = np.sum(np.average(np.square(au_f), axis=0))
                lk = -0.691 + 10*np.log10(z)
                Lk, Z = np.append(Lk, lk), np.append(Z, z)
        elif au.ndim == 1:
            if cut_start:
                au = au[0: int(sr*cut_start)]
            q1, q2 = divmod(au.shape[0], hop)
            q3 = step - hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros(q3), axis=0)
            for i in range(0, q1):
                au_f = au[i*hop: i*hop+step]
                au_f = self.prefilter(au_f)
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

def change_LR_peak_ratio(au, scale):
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

