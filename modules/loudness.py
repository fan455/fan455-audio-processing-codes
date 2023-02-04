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
# To normalize the integrated lufs, use the 'Ilufs_meter().norm()' function. To normalize the short-term or momentary lufs, use the 'Mlufs_meter().norm()' function.
# Both mono and stereo input audio arrays with amplitudes between -1 and 1 are supported.

import numpy as np
from scipy import signal

def amp2db(amp: float): # zero or positive amp value range between 0 and 1.
    return 20*np.log10(amp)

def db2amp(db: float): # zero or negative db value.
    return np.power(10, db/20)

def change_vol(au, db_change):
    return au*db2amp(db_change)

def check_clipping(au):
    assert np.amax(np.abs(au)) >= 1, 'Clipping has occurred.'
    
def re_compare(au, au_re, sr):
    print('reconstruction comparison:')
    if sr:
        print(f'difference in length: {round((au_re.shape[0] - au.shape[0])/self.sr, 4)} seconds')
    if au.ndim == 2:
        print(f'max error: {round(amp2db(np.amax(np.abs(au_re[:au.shape[0], :] - au))), 4)}db')
    elif au.ndim == 1:
        print(f'max error: {round(amp2db(np.amax(np.abs(au_re[:au.shape[0]] - au))), 4)}db')

class mlufs_meter():
    # This allows the pre-computation of prefilter coefficients for faster response, particularly when batch processing.
    def __init__(self, sr, T=0.4, overlap=0.75, threshold=-70.0):
        """
        sr: float (Hz). Sample rate for audio. If you want to process different sample rates, you need to set more than 1 meters.
        T: float (seconds). Time length of each window. You can change it to 3 to get the short-term lufs (Slufs).
        overlap: float (fraction). Proportion of overlapping between windows.
        cut_start: float (seconds). The start seconds to only analyze.
        threshold: float (LUFS or LKFS). If the LUFS is lower than this threshold, the meter will return -inf instead of very big negative numbers for runtime stability.
        Only works for mono or stereo audio because I just summed all the channels and didn't calculate the different weights in case of a 5-channel audio input.
        """
        self.sr, self.T, self.overlap, self.threshold = sr, T, overlap, threshold
        self.z_threshold = np.power(10, (self.threshold+0.691)/10)
        self.step, self.hop = int(self.sr*self.T), int(self.sr*self.T*(1-self.overlap))
        if self.sr == 48000:
            # coefficients in the ITU documentation.
            b0_1, b1_1, b2_1 = 1.53512485958697, -2.69169618940638, 1.19839281085285
            a0_1, a1_1, a2_1 = 1.0 , -1.69065929318241, 0.73248077421585
            b0_2, b1_2, b2_2 = 1.0, -2.0, 1.0
            a0_2, a1_2, a2_2 = 1.0, -1.99004745483398, 0.99007225036621
        elif self.sr == 44100:
            # coefficients calculation by BrechtDeMan, super close to the ITU documentation.
            b0_1, b1_1, b2_1 = 1.5308412300498355, -2.6509799951536985, 1.1690790799210682
            a0_1, a1_1, a2_1 = 1.0, -1.6636551132560204, 0.7125954280732254
            b0_2, b1_2, b2_2 = 1.0, -2.0, 1.0
            a0_2, a1_2, a2_2 = 1.0, -1.9891696736297957, 0.9891990357870394
        else:
            # coefficients calculation by BrechtDeMan, super close to the ITU documentation. 
            # pre-filter 1
            f0 = 1681.9744509555319
            G  = 3.99984385397
            Q  = 0.7071752369554193
            K  = np.tan(np.pi * f0 / sr) 
            Vh = np.power(10.0, G / 20.0)
            Vb = np.power(Vh, 0.499666774155)
            a0_1_ = 1.0 + K / Q + K * K
            b0_1 = (Vh + Vb * K / Q + K * K) / a0_1_
            b1_1 = 2.0 * (K * K -  Vh) / a0_1_
            b2_1 = (Vh - Vb * K / Q + K * K) / a0_1_
            a0_1 = 1.0
            a1_1 = 2.0 * (K * K - 1.0) / a0_1_
            a2_1 = (1.0 - K / Q + K * K) / a0_1_
            # pre-filter 2
            f0 = 38.13547087613982
            Q  = 0.5003270373253953
            K  = np.tan(np.pi * f0 / sr)
            a0_2 = 1.0
            a1_2 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
            a2_2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
            b0_2 = 1.0
            b1_2 = -2.0
            b2_2 = 1.0
            
        self.sos = np.array([[b0_1, b1_1, b2_1, a0_1, a1_1, a2_1], \
                            [b0_2, b1_2, b2_2, a0_2, a1_2, a2_2]])
            
    def prefilter(self, au):
        au = signal.sosfilt(self.sos, au, axis=0)
        return au

    def pad_zeros(self, au):
        # Since the output of mlufs will be affected by zeros padding, this is for comparison purpose.
        if au.ndim == 2:
            nchannel = au.shape[-1]
            q1, q2 = divmod(au.shape[0], self.hop)
            q3 = self.step - self.hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros((q3, nchannel)), axis=0)
                print(f'{round(q3/self.sr, 4)} seconds of silence padded at the end of audio.')
            else:
                print('no padding')
        elif au.ndim == 1:
            q1, q2 = divmod(au.shape[0], self.hop)
            q3 = self.step - self.hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros(q3), axis=0)
                print(f'{round(q3/self.sr, 4)} seconds of silence padded at the end of audio.')
            else:
                print('no padding')
        else:
            raise ValueError(f'au.ndim = {au.ndim} is not supported.')
        return au
    
    def get(self, au, cut_start=None):
        # Get the full Mlufs array corresponding to the windowed audio. The audio array will be padded zeros at the end to complete the last window.
        Mlufs = np.empty(0)
        if au.ndim == 2:
            if cut_start:
                au = au[0: int(self.sr*cut_start), :]
            q1, q2 = divmod(au.shape[0], self.hop)
            q3 = self.step - self.hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros((q3, au.shape[-1])), axis=0)
            for i in range(0, q1):
                au_f = au[i*self.hop: i*self.hop+self.step, :]
                au_f = self.prefilter(au_f)
                z = np.sum(np.average(np.square(au_f), axis=0))
                if z < self.z_threshold:
                    mlufs = float('-inf')
                else:
                    mlufs = -0.691 + 10*np.log10(z)
                Mlufs = np.append(Mlufs, mlufs)
        elif au.ndim == 1:
            if cut_start:
                au = au[0: int(self.sr*cut_start)]
            q1, q2 = divmod(au.shape[0], self.hop)
            q3 = self.step - self.hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros(q3), axis=0)
            for i in range(0, q1):
                au_f = au[i*self.hop: i*self.hop+self.step]
                au_f = self.prefilter(au_f)
                z = np.average(np.square(au_f))
                if z < self.z_threshold:
                    mlufs = float('-inf')
                else:
                    mlufs = -0.691 + 10*np.log10(z)
                Mlufs = np.append(Mlufs, mlufs)
        else:
            raise ValueError(f'au.ndim = {au.ndim} is not supported.')
        return Mlufs

    def get_max(self, au, cut_start=None):
        # Get the maxinum momentary lufs.
        return np.amax(self.get(au, cut_start=cut_start))

    def display_max(self, au, cut_start=None):
        # print the maxinum momentary lufs.
        mlufs_max = np.amax(self.get(au, cut_start=cut_start))
        print(f'mlufs_max = {round(mlufs_max, 4)} LUFS')

    def norm(self, au, target=-20.0, cut_start=None):
        # Normalize the maxinum momentary lufs.
        return au*db2amp(target - self.get_max(au, cut_start=cut_start))
    
class ilufs_meter():
    # This allows the pre-computation of prefilter coefficients for faster response, particularly when batch processing.
    def __init__(self, sr, T=0.4, overlap=0.75, threshold=-70.0):
        """
        sr: float (Hz). Sample rate for audio. If you want to process different sample rates, you need to set more than 1 meters.
        T: float (seconds). Time length of each window. It means the same as 'gating' in the ITU doc.
        overlap: float (fraction). Proportion of overlapping between windows.
        threshold: float (LUFS or LKFS). If the LUFS is lower than this threshold, the meter will return -inf instead of very big negative numbers for runtime stability.
        Only works for mono or stereo audio because I just summed all the channels and didn't calculate the different weights in case of a 5-channel audio input.
        """
        self.sr, self.T, self.overlap, self.threshold = sr, T, overlap, threshold
        self.z_threshold = np.power(10, (self.threshold+0.691)/10)
        self.step, self.hop = int(self.sr*self.T), int(self.sr*self.T*(1-self.overlap))
        if self.sr == 48000:
            # coefficients in the ITU documentation.
            b0_1, b1_1, b2_1 = 1.53512485958697, -2.69169618940638, 1.19839281085285
            a0_1, a1_1, a2_1 = 1.0 , -1.69065929318241, 0.73248077421585
            b0_2, b1_2, b2_2 = 1.0, -2.0, 1.0
            a0_2, a1_2, a2_2 = 1.0, -1.99004745483398, 0.99007225036621
        elif self.sr == 44100:
            # coefficients calculation by BrechtDeMan, super close to the ITU documentation.
            b0_1, b1_1, b2_1 = 1.5308412300498355, -2.6509799951536985, 1.1690790799210682
            a0_1, a1_1, a2_1 = 1.0, -1.6636551132560204, 0.7125954280732254
            b0_2, b1_2, b2_2 = 1.0, -2.0, 1.0
            a0_2, a1_2, a2_2 = 1.0, -1.9891696736297957, 0.9891990357870394
        else:
            # coefficients calculation by BrechtDeMan, super close to the ITU documentation. 
            # pre-filter 1
            f0 = 1681.9744509555319
            G  = 3.99984385397
            Q  = 0.7071752369554193
            K  = np.tan(np.pi * f0 / sr) 
            Vh = np.power(10.0, G / 20.0)
            Vb = np.power(Vh, 0.499666774155)
            a0_1_ = 1.0 + K / Q + K * K
            b0_1 = (Vh + Vb * K / Q + K * K) / a0_1_
            b1_1 = 2.0 * (K * K -  Vh) / a0_1_
            b2_1 = (Vh - Vb * K / Q + K * K) / a0_1_
            a0_1 = 1.0
            a1_1 = 2.0 * (K * K - 1.0) / a0_1_
            a2_1 = (1.0 - K / Q + K * K) / a0_1_
            # pre-filter 2
            f0 = 38.13547087613982
            Q  = 0.5003270373253953
            K  = np.tan(np.pi * f0 / sr)
            a0_2 = 1.0
            a1_2 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
            a2_2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
            b0_2 = 1.0
            b1_2 = -2.0
            b2_2 = 1.0
            
        self.sos = np.array([[b0_1, b1_1, b2_1, a0_1, a1_1, a2_1], \
                            [b0_2, b1_2, b2_2, a0_2, a1_2, a2_2]])
            
    def prefilter(self, au):
        au = signal.sosfilt(self.sos, au, axis=0)
        return au

    def pad_zeros(self, au):
        # Since the output of ilufs will be affected by zeros padding, this is for comparison purpose.
        if au.ndim == 2:
            nchannel = au.shape[-1]
            q1, q2 = divmod(au.shape[0], self.hop)
            q3 = self.step - self.hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros((q3, nchannel)), axis=0)
                print(f'{round(q3/self.sr, 4)} seconds of silence padded at the end of audio.')
            else:
                print('no padding')
        elif au.ndim == 1:
            q1, q2 = divmod(au.shape[0], self.hop)
            q3 = self.step - self.hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros(q3), axis=0)
                print(f'{round(q3/self.sr, 4)} seconds of silence padded at the end of audio.')
            else:
                print('no padding')
        else:
            raise ValueError(f'au.ndim = {au.ndim} is not supported.')
        return au

    def get(self, au, cut_start=None):
        # Get the integrated lufs of a mono or stereo audio input. The audio array will be padded zeros at the end to complete the last window.
        Mlufs, Z = np.empty(0), np.empty(0)
        if au.ndim == 2:
            if cut_start:
                au = au[0: int(self.sr*cut_start), :]
            nchannel = au.shape[-1]
            q1, q2 = divmod(au.shape[0], self.hop)
            q3 = self.step - self.hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros((q3, nchannel)), axis=0)       
            for i in range(0, q1):
                au_f = au[i*self.hop: i*self.hop+self.step, :]
                au_f = self.prefilter(au_f)
                z = np.sum(np.average(np.square(au_f), axis=0))
                if z < self.z_threshold:
                    mlufs = float('-inf')
                else:
                    mlufs = -0.691 + 10*np.log10(z)
                Mlufs, Z = np.append(Mlufs, mlufs), np.append(Z, z)
        elif au.ndim == 1:
            if cut_start:
                au = au[0: int(self.sr*cut_start)]
            q1, q2 = divmod(au.shape[0], self.hop)
            q3 = self.step - self.hop - q2
            if q3 > 0:
                au = np.append(au, np.zeros(q3), axis=0)
            for i in range(0, q1):
                au_f = au[i*self.hop: i*self.hop+self.step]
                au_f = self.prefilter(au_f)
                z = np.average(np.square(au_f), axis=0)
                if z < self.z_threshold:
                    mlufs = float('-inf')
                else:
                    mlufs = -0.691 + 10*np.log10(z)
                Mlufs, Z = np.append(Mlufs, mlufs), np.append(Z, z)
        else:
            raise ValueError(f'au.ndim = {au.ndim} is not supported.')
        Z0 = Z[Mlufs > -70.0]
        if Z0.size == 0:
            ilufs = float('-inf')
        else:
            z1 = np.average(Z0)
            if z1 >= self.z_threshold:
                Z = Z[Mlufs > -0.691 + 10*np.log10(z1) - 10]
            else:
                pass 
        if Z.size == 0:
            ilufs = float('-inf')
        else:
            z2 = np.average(Z)
            if z2 >= self.z_threshold:
                ilufs = -0.691 + 10*np.log10(z2)
            else:
                ilufs = float('-inf')
        return ilufs

    def display(self, au, cut_start=None):
        # print the maxinum momentary lufs.
        ilufs = self.get(au, cut_start=cut_start)
        print(f'ilufs = {round(ilufs, 4)} LUFS')

    def norm(self, au, target=-23.0, cut_start=None):
        return au*db2amp(target - self.get(au, cut_start=cut_start))

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
