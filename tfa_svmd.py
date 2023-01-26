# Sequential Variational Mode Decomposition (SVMD)
# This algorithm is developed by Wei Chen from JiangXi University of Finance and Economics.
# Reference paper:
"""
@ARTICLE{2021arXiv210305874C,
       author = {{Chen}, Wei},
        title = "{A Sequential Variational Mode Decomposition Method}",
      journal = {arXiv e-prints},
     keywords = {Electrical Engineering and Systems Science - Signal Processing},
         year = 2021,
        month = mar,
          eid = {arXiv:2103.05874},
        pages = {arXiv:2103.05874},
archivePrefix = {arXiv},
       eprint = {2103.05874},
 primaryClass = {eess.SP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210305874C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""
# This is a python implementation of the SVMD algorithm. I have no copyright regarding this python code.
# The end effect handling algorithm has not been implemented.

import timeit
import numpy as np
import scipy.fft

def abs2(x):
    # Avoid square root calculation.
    #return np.square(x.real) + np.square(x.imag)
    return x.real**2 + x.imag**2 # This line seems faster than the above line.

def svmd(y, out_thr=1e-5, in_thr=1e-10, out_iter_max=9, in_iter_max=50, alpha=1, beta=1e-2, return_type='modes'):
    """
    Parameters:
    y: 1d real array. The input signal array, need to be 1d, real, and better within range [-1, 1].
    out_thr: positive float. The threshold for outer iterations. A smaller value may result in more modes decomposed.
    in_thr: positive float. The threshold for inner iterations. A smaller value may result in more accurate modes decomposed.
    out_iter_max: positive int. Maxinum outer iteration times. It can avoid endless iteration case.
    in_iter_max: positive int. Maxinum inner iteration times. It can avoid endless iteration case.
    alpha: positive float. Penalty coffecient for the second quadratic term in the optimization.
    beta: positive float. Penalty coffecient for the third quadratic term in the optimization.
    return_type: str, 'modes' or 'modes, residual'. If 'modes', return y_modes array including residual at index [-1, :].
        If 'residual', return tuple (y_modes, y_res).
        
    Returns (depending on return_type):
    Modes: nd real array. The decomposed modes of y of shape (number of modes, size of y), excluding or including the residual.
    y_res: The residue of input after subtracting previous Modes.
    """
    print('SVMD started.', '\n')
    start_time = timeit.default_timer()
    print('Input information:')
    assert y.ndim == 1, 'y.ndim = {y.ndim}'
    y_size = y.size
    print(f'y.size = {y_size}')
    if y_size % 2 != 0:
        input_size_is_odd = True # no adjustment.
    else:
        input_size_is_odd = False
        y = np.append(y, 0.0)
        # make input size odd because even fft size will result in a frequency that is both positive and negative.
        print('The input is padded 1 zero at the end because its size is even.')
    print(f'input_size_is_odd = {input_size_is_odd}', '\n')

    print('Decomposition information:')
    z = 2*scipy.fft.rfft(y, axis=0, norm='backward') # transform input to frequency domain. z represents complex.
    print(f'z.size = {z.size}', '\n')
    f = np.arange(z.size)
    Modes = []
    
    for k in range(1, out_iter_max+1):
        f0 = np.argmax(np.abs(z))
        mode_prev = np.zeros(z.size, dtype=np.complex128)
        #mode_prev[f0] = abs(z[f0])
        mode_prev[f0] = z[f0] # I'm not sure if this line or the above line is correct.
            
        for i in range(1, in_iter_max+1):
            mode_prev_sq = abs2(mode_prev)
            fc = np.sum(f*mode_prev_sq)/np.sum(mode_prev_sq)
            z_prev_sq = abs2(z - mode_prev)
            fc_res = np.sum(f*z_prev_sq)/np.sum(z_prev_sq)
            mode_next = (z*(1+beta*np.square(f-fc_res)))/(1+alpha*np.square(f-fc)+beta*np.square(f-fc_res))
            #if np.sum(np.square(np.abs(mode_next)-np.abs(mode_prev))) > in_thr:
            if np.sum(abs2(mode_next-mode_prev)) > in_thr: # I'm not sure if this line or the above line is correct.
                mode_prev = mode_next.copy()
            else:
                break

        print(f'The {k}th outer iteration took {i} inner iterations.')
        print(f'mode {k}: f0 = {f0}, fc = {round(fc, 2)}', '\n')
        Modes.append(mode_next)
        z -= mode_next
        if np.sum(abs2(z)) <= out_thr:
            break
        
    print(f'Totally {k} modes extracted (excluding residual).')
    Modes.append(z)
    Modes = np.append(np.array(Modes), np.zeros((k+1, y_size//2)), axis=1)
    Modes = np.real(scipy.fft.ifft(Modes, axis=1, norm='backward')) # transform output back to time domain.
    if not input_size_is_odd:
        Modes = np.delete(Modes, -1, axis=1) # delete the last element of output to compensate.
    print('The last element of output is deleted because input size is even.', '\n')
    end_time = timeit.default_timer()
    print(f'SVMD completed, running time: {round((end_time-start_time), 4)} seconds.')
    if return_type == 'modes':
        return Modes
    elif return_type == 'modes, residual':
        return Modes[:-1, :], Modes[-1, :]
    else:
        raise ValueError(f'return_type "{return_type}" is not supported.')

def svmd_refined(y, out_thr=1e-5, in_thr=1e-10, out_iter_max=9, in_iter_max=50, alpha=1, beta=1e-2, merge_range=1.5, return_type='modes'):
    """
    SVMD including a refinement process to determine the mode number and merge modes with close center frequencies.
    You will be prompted a input requirement to determine the mode number after you have analyzed the normalized distances between modes.

    Parameters: (additional)
    merge_range: positive float. Determine the lower and upper limit of merge range on a multiplication basis.
        For example, for mode_i (i<=mode number) with center frequency of 15Hz and merge range 1.5, only modes with center frequencies
        in range (10, 22.5)Hz can be merged into mode_i.
    """
    from plot import plot
    print('SVMD started.', '\n')
    start_time = timeit.default_timer()
    print('Input information:')
    assert y.ndim == 1, 'y.ndim = {y.ndim}'
    y_size = y.size
    print(f'y.size = {y_size}')
    if y_size % 2 != 0:
        input_size_is_odd = True # no adjustment.
    else:
        input_size_is_odd = False
        y = np.append(y, 0.0)
        # make input size odd because even fft size will result in a frequency that is both positive and negative.
        print('The input is padded 1 zero at the end because its size is even.')
    print(f'input_size_is_odd = {input_size_is_odd}', '\n')

    print('Decomposition information:')
    z = 2*scipy.fft.rfft(y, axis=0, norm='backward') # transform input to frequency domain. z represents complex.
    print(f'z.size = {z.size}', '\n')
    f = np.arange(z.size)
    Modes, Fc, Distances = [], [], []
    
    for k in range(1, out_iter_max+1):
        f0 = np.argmax(np.abs(z))
        mode_prev = np.zeros(z.size, dtype=np.complex128)
        #mode_prev[f0] = abs(z[f0])
        mode_prev[f0] = z[f0] # I'm not sure if this line or the above line is correct.
            
        for i in range(1, in_iter_max+1):
            mode_prev_sq = abs2(mode_prev)
            fc = np.sum(f*mode_prev_sq)/np.sum(mode_prev_sq)
            z_prev_sq = abs2(z - mode_prev)
            fc_res = np.sum(f*z_prev_sq)/np.sum(z_prev_sq)
            mode_next = (z*(1+beta*np.square(f-fc_res)))/(1+alpha*np.square(f-fc)+beta*np.square(f-fc_res))
            #if np.sum(np.square(np.abs(mode_next)-np.abs(mode_prev))) > in_thr:
            if np.sum(abs2(mode_next-mode_prev)) > in_thr: # I'm not sure if this line or the above line is correct.
                mode_prev = mode_next.copy()
            else:
                break

        print(f'The {k}th outer iteration took {i} inner iterations.')
        print(f'mode {k}: f0 = {f0}, fc = {round(fc, 2)}', '\n')
        Modes.append(mode_next)
        Fc.append(fc)
        z -= mode_next
        if np.sum(abs2(z)) <= out_thr:
            break
        
    print(f'Totally {k} modes extracted (excluding residual).')
    Modes.append(z)
    Modes = np.append(np.array(Modes), np.zeros((k+1, y_size//2)), axis=1)
    Modes = np.real(scipy.fft.ifft(Modes, axis=1, norm='backward')) # transform output back to time domain.
    if not input_size_is_odd:
        Modes = np.delete(Modes, -1, axis=1) # delete the last element of output to compensate.
    print('The last element of output is deleted because input size is even.', '\n')

    print('Refinement process', '\n')
    norm = (0.5/np.amax(np.abs(Modes), axis=1)).reshape((k+1, 1))
    for i in range(1, k):
        distance = np.amin(np.sqrt(np.average(np.square(norm[:i,:]*Modes[:i,:] - norm[i,0]*Modes[i,:]), axis=1)))
        Distances.append(distance)
    Distances = np.array(Distances)
    print('Please input after closing this figure:')
    print(f'(Tip: Choose the mode number after which the distance sharply drops closer to zero.)')
    plot(Distances, np.arange(2, k+1), x_label='mode number', y_label='minimum normalized distance')
    nMode = int(input(f'Number of modes (should be less than {k}): '))

    Fc = np.array(Fc)
    Fc_main = np.broadcast_to(Fc[:nMode].reshape((nMode,1)), (nMode, k-nMode))
    Fc_other = np.broadcast_to(Fc[nMode:].reshape((1, k-nMode)), (nMode, k-nMode))
    Merge = np.argmin(np.abs(Fc_main-Fc_other), axis=0)
    Merge[Fc[nMode:] < Fc[Merge]/merge_range] = k
    Merge[Fc[nMode:] > Fc[Merge]*merge_range] = k    
    for i in range(nMode, k):
        merge = Merge[i-nMode]
        Modes[merge] += Modes[i]
        if merge != k:
            print(f'mode {i+1} is merged into mode {merge+1}.')
        else:
            print(f'mode {i+1} is merged into the residual.')
    Modes = np.delete(Modes, np.arange(nMode, k), axis=0)
    
    end_time = timeit.default_timer()
    print(f'SVMD completed, running time: {round((end_time-start_time), 4)} seconds.')
    if return_type == 'modes':
        return Modes
    elif return_type == 'modes, residual':
        return Modes[:-1, :], Modes[-1, :]
    else:
        raise ValueError(f'return_type "{return_type}" is not supported.')
