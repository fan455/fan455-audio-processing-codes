import numpy as np

def trim_leading_silence(au, sr, amp, du, lead_du=None):
    """
    Trim the leading silence of an audio given the maxinum loudness (amp) and mininum duration (du) of silence.

    au: ndarray. Input audio array of 1 or 2 dimensions.
    sr: int. Sample rate of the input audio.
    amp: float (between 0 and 1). abs(au) <= amp is silence if the duration requirement is met.
    du: float (seconds). Minimum consecutive seconds for low amp values to be recognized as silence.
    lead_du: float (seconds). Seconds at the beginning to analyze. If set to None, use the whole audio.
    """
    n = int(sr*du)
    size = au.shape[0]
    if lead_du == None:
        lead = size
    else:
        lead = int(sr*lead_du)
    k = 0
    if au.ndim == 1:
        bool_arr = np.array(abs(au[0: lead]) - amp > 0) # True is sound. False is silence. 
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
    elif au.ndim == 2:
        bool_arr = np.array(abs(au[0: lead, :]) - amp > 0) # True is sound. False is silence.
        while np.any(bool_arr[k*n: (k+1)*n, :]) == False:
            k += 1
            if (k+1)*n > size:
                raise ValueError('Unable to trim. Probably due to the lack of sound in the lead_du range.')
        trim = k*n
        if trim == 0:
            #print('not trimmed')
            return au
        else:
            #print(f'trimmed {round(trim/sr, 4)} seconds')
            return au[trim:, :]
    else:
        raise ValueError('audio needs to have 1 or 2 dimensions. Maybe your audio array is framed?')
