import numpy as np

def amp2db(amp: float): # zero or positive amp value range between 0 and 1.
    return 20*np.log10(amp)

def db2amp(db: float): # zero or negative db value.
    return np.power(10, db/20)

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

def normalize_mid(au, amp=0.35):
    """
    Normalize the peak volume of the mid channel of a stereo wav file under the normal -3db pan law.
    input array, output array, no read or write audio files.
    """
    au *= 2*amp/np.amax(np.abs(np.sum(au, axis=-1)))
    return au

def normalize_mono(au, amp=0.5):
    """
    Normalize the peak volume of a mono audio.
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
