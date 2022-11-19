import numpy as np
from loudness import amp2db, db2amp

def trim_start(au, sr, du):
    """
    trim_du: float, seconds. Starting time duration to trim.
    """
    if au.ndim == 2:
        return au[int(np.floor(sr*du)):, :]
    elif au.ndim == 1:
        return au[int(np.floor(sr*du)):]
    else:
        raise ValueError('au.ndim != 1 or 2')

def trim_end(au, sr, du):
    """
    trim_du: float, seconds. Ending time duration to trim.
    """
    if au.ndim == 2:
        return au[:-int(np.floor(sr*du)), :]
    elif au.ndim == 1:
        return au[:-int(np.floor(sr*du))]
    else:
        raise ValueError('au.ndim != 1 or 2')

def trim_start_end(au, sr, du1, du2):
    """
    du1: float, seconds. Starting time duration to trim.
    du2: float, seconds. Ending time duration to trim.
    """
    idx_1, idx_2 = int(np.floor(sr*du1)), -int(np.floor(sr*du2))
    if idx_2 != 0:
        pass
    else:
        idx_2 = None
    if au.ndim == 2:
        return au[idx_1: -idx_2, :]
    elif au.ndim == 1:
        return au[idx_1: -idx_2]
    else:
        raise ValueError('au.ndim != 1 or 2')

def split2_du1(au, sr, du1):
    """
    Split an audio array into 2 arrays by specifying the duration of the first split array.
    du1: float, seconds. Duration of the first split array. Will be rounded ceil to index.
    """
    idx = int(np.ceil(sr*du1))
    if au.ndim == 2:
        return au[: idx, :], au[idx:, :]
    elif au.ndim == 1:
        return au[: idx], au[idx:]
    else:
        raise ValueError('au.ndim != 1 or 2')

def split2_du2(au, sr, du2):
    """
    Split an audio array into 2 arrays by specifying the duration of the second split array.
    du2: float, seconds. Duration of the second split array. Will be rounded ceil to index.
    """
    idx = int(np.floor(sr*du2))
    if au.ndim == 2:
        return au[: idx, :], au[idx:, :]
    elif au.ndim == 1:
        return au[: idx], au[idx:]
    else:
        raise ValueError('au.ndim != 1 or 2')
    
class onset_trimmer():
    import librosa.onset
    from channel import sum2
    def __init__(self, sr, T=0.02, sum_channels=False, lead_du=None, cut_win_du=0.1):
        self.sr, self.T, self.sum_channels, self.lead_du = sr, T, sum_channels, lead_du
        self.hop = int(self.sr*self.T)
        if lead_du == None:
            self.lead_size = au.shape[0]
        else:
            self.lead_size = int(sr*self.lead_du)
            self.cut_win_size = int(self.sr*cut_win_du)
            self.cut_env = np.append(np.ones(self.lead_size-self.cut_win_size), np.hanning(self.cut_win_size))
    
    def get_lead(self, au_mono):
        if self.lead_du == None:
            return au_mono[0: self.lead_size]
        else:
            return au_mono[0: self.lead_size]*self.cut_env  

    #def get_onset_env(self, lead):
        #return librosa.onset.onset_strength(y=lead, sr=self.sr, )

    def get_onset(self, lead):
        return librosa.onset.onset_detect(y=lead, sr=self.sr, units='samples', hop_length=self.hop, backtrack=True)

    def trim(self, au):
        if au.ndim == 2 and au.shape[1] == 2:
            if self.sum_channels == False:
                lead1, lead2 = self.get_lead(au[:, 0]), self.get_lead(au[:, 1])
                trim1, trim2 = self.get_onset(lead1)[0], self.get_onset(lead2)[0]
                trim = min(trim1, trim2)
            else:
                lead = self.get_lead(sum2(au))
                trim = self.get_onset(lead)[0]
            print(f'trimmed {round(trim/self.sr, 4)} seconds')
            return au[trim:, :]
        elif au.ndim == 1:
            lead = self.get_lead(au)
            trim = self.get_onset(lead)[0]
            print(f'trimmed {round(trim/self.sr, 4)} seconds')
            return au[trim:]
        else:
            raise ValueError('audio needs to be unframed mono or stereo.')
  
def trim_ls2(au, sr, db, diff, T, lead_du=None):
    """
    trim leading silence, depending on the loudness of the moving window relative to the first window (usually silence).
    db: float, decibels.
    diff: float, decibels.
    """
    n, amp, scale = int(sr*T), db2amp(db), db2amp(diff)
    if lead_du == None:
        lead_size = au.shape[0]
    else:
        lead_size = int(sr*lead_du)
    if au.ndim == 2:
        lead = np.abs(au[0: lead_size, :])
        k = 1
        M0, M1 = np.amax(lead[0: k*n, :], axis=0), np.amax(lead[n: 2*n, :], axis=0)
        if np.any(M0 > amp) == True:
            print('not trimmed')
            return au
        else:
            while np.any(M1/M0 > scale) == False:
                k += 1
                if (k+1)*n > lead_size:
                    #print('not trimmed')
                    return au
                    break
                else:
                    M0 = M1
                    M1 = np.amax(lead[k*n: (k+1)*n, :], axis=0)
            trim = int((k-1)*n)
            print(f'trimmed {round(trim/sr, 4)} seconds')
            return au[trim:, :]
    elif au.ndim == 1:
        lead = np.abs(au[0: lead_size])
        k = 1
        M0, M1 = np.amax(lead[0: n]), np.amax(lead[n: 2*n] )
        if M0 > amp:
            #print('not trimmed')
            return au
        else:
            while M1/M0 <= scale:
                k += 1
                if (k+1)*n > lead_size:
                    #print('not trimmed')
                    return au
                    break
                else:
                    M0 = M1
                    M1 = np.amax(lead[k*n: (k+1)*n])
            trim = int((k-1)*n)
            #print(f'trimmed {round(trim/sr, 4)} seconds')
            return au[trim:]
    else:
        raise ValueError('audio needs to have 1 or 2 dimensions. Maybe your audio array is framed?')

def trim_ls(au, sr, db, T, lead_du=None):
    """
    Trim the leading silence of an audio given the maxinum loudness (amp) and mininum duration (du) of silence.

    au: ndarray. Input audio array of 1 or 2 dimensions.
    sr: int. Sample rate of the input audio.
    db: float (0 to -inf). abs(au) <= db2amp(db) is silence if the duration requirement is met.
    du: float (seconds). Minimum consecutive seconds for low amp values to be recognized as silence.
    lead_du: float (seconds). Seconds at the beginning to analyze. If set to None, use the whole audio.
    """
    n, amp = int(sr*T), db2amp(db)
    if lead_du == None:
        lead_size = au.shape[0]
    else:
        lead_size = int(sr*lead_du)
    k = 0
    if au.ndim == 2:
        bool_arr = np.array(abs(au[0: lead_size, :]) - amp > 0) # True is sound. False is silence.
        while np.any(bool_arr[k*n: (k+1)*n, :]) == False:
            k += 1
            if (k+1)*n > lead_size:
                raise ValueError('Unable to trim. Probably due to the lack of sound in the lead_du range.')
        trim = k*n
        if trim == 0:
            print('not trimmed')
            return au
        else:
            print(f'trimmed {round(trim/sr, 4)} seconds')
            return au[trim:, :]
    elif au.ndim == 1:
        bool_arr = np.array(abs(au[0: lead_size]) - amp > 0) # True is sound. False is silence. 
        while np.any(bool_arr[k*n: (k+1)*n]) == False:
            k += 1
            if (k+1)*n > size:
                raise ValueError('Unable to trim. Probably due to the lack of sound in the lead_du range.')
        trim = k*n
        if trim == 0:
            #print('not trimmed')
            return au
        else:
            #print(f'trimmed {round(trim/sr, 4)} seconds')
            return au[trim:]
    else:
        raise ValueError('audio needs to have 1 or 2 dimensions. Maybe your audio array is framed?')
