import numpy as np

def trim_ls2(au, sr, amp, scale, du, lead_du=None):
    """
    trim leading silence, depending on the loudness of the moving window relative to the first window (usually silence).
    """
    n = int(sr*du)
    if lead_du == None:
        lead_size = au.shape[0]
    else:
        lead_size = int(sr*lead_du)  
    if au.ndim == 1:
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
                    M1 = np.amax(lead[k*n: (k+1)*n])
            trim = int((k-0.5)*n)
            #print(f'trimmed {round(trim/sr, 4)} seconds')
            return au[trim:]
    elif au.ndim == 2:
        lead = np.abs(au[0: lead_size, :])
        k = 1
        M0, M1 = np.amax(lead[0: k*n, :], axis=0), np.amax(lead[n: 2*n, :], axis=0)
        if np.any(M0 > amp) == True:
            #print('not trimmed')
            return au
        else:
            while np.any(M1/M0 > scale) == False:
                k += 1
                if (k+1)*n > lead_size:
                    #print('not trimmed')
                    return au
                    break
                else:
                    M1 = np.amax(lead[k*n: (k+1)*n, :], axis=0)
            trim = int((k-0.5)*n)
            #print(f'trimmed {round(trim/sr, 4)} seconds')
            return au[trim:, :]
    else:
        raise ValueError('audio needs to have 1 or 2 dimensions. Maybe your audio array is framed?')

def trim_ls(au, sr, amp, du, lead_du=None):
    """
    Trim the leading silence of an audio given the maxinum loudness (amp) and mininum duration (du) of silence.

    au: ndarray. Input audio array of 1 or 2 dimensions.
    sr: int. Sample rate of the input audio.
    amp: float (between 0 and 1). abs(au) <= amp is silence if the duration requirement is met.
    du: float (seconds). Minimum consecutive seconds for low amp values to be recognized as silence.
    lead_du: float (seconds). Seconds at the beginning to analyze. If set to None, use the whole audio.
    """
    n = int(sr*du)
    if lead_du == None:
        lead_size = au.shape[0]
    else:
        lead_size = int(sr*lead_du)
    k = 0
    if au.ndim == 1:
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
    elif au.ndim == 2:
        bool_arr = np.array(abs(au[0: lead_size, :]) - amp > 0) # True is sound. False is silence.
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
