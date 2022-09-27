import numpy as np

def amp2db(amp: float): # zero or positive amp value range between 0 and 1.
    return 20*np.log10(amp)

def db2amp(db: float): # zero or negative db value.
    return np.power(10, db/20)

def get_peak(au):
    au_abs = np.abs(au)
    peak_amp = np.amax(au_abs)
    peak_db = amp2db(peak_amp)
    print(f'peak_amp = {round(peak_amp, 5)}, peak_db = {round(peak_db, 2)}')
