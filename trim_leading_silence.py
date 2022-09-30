import numpy as np

def trim_leading_silence(au, sr, amp, du, lead_du=None):
    """
    amp: float. abs(au) <= amp is silence.
    lead_du: float. Seconds at the beginning to analyze. If set to None, use the whole audio.
    """
    n = int(sr*du)
    size = au.shape[0]
    if lead_du == None:
        lead = size
    else:
        lead = int(sr*lead_du)
    if au.ndim == 1:
        bool_mono = np.array(abs(au[0: lead, 0]) - amp > 0) # True is sound. False is silence.
        k = 0
        while np.any(bool_mono[k*n: (k+1)*n]) == False:
            k += 1
        trim = k*n
        if trim == 0:
            #print('not trimmed')
            return au
        elif trim < size:
            #print(f'trimmed {round(trim/sr, 4)} seconds')
            return au[trim:]
        else:
            raise ValueError('Unable to trim. Probably due to the lack of sound in the lead_du range.')
    elif au.ndim == 2:
        bool_L = np.array(abs(au[0: lead, 0]) - amp > 0) # True is sound. False is silence.
        bool_R = np.array(abs(au[0: lead, 1]) - amp > 0)
        k = 0
        while np.any(bool_L[k*n: (k+1)*n]) == False:
            k += 1
        trim_L = k*n
        k = 0
        while np.any(bool_R[k*n: (k+1)*n]) == False:
            k += 1
        trim_R = k*n
        trim = min(trim_L, trim_R)
        if trim == 0:
            #print('not trimmed')
            return au
        elif trim < size:
            #print(f'trimmed {round(trim/sr, 4)} seconds')
            return au[trim:, :]
        else:
            raise ValueError('Unable to trim. Probably due to the lack of sound in the lead_du range.')
    else:
        raise ValueError('Only support mono or stereo audio array input')
