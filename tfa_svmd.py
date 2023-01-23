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

def svmd(y, out_thr=1e-5, in_thr=1e-10, out_iter_max=20, in_iter_max=50, alpha=2.5e+1, beta=1e-1, return_type='modes'):
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
    y_modes: nd real array. The decomposed modes of y of shape (number of modes, size of y), excluding or including residual.
    y_res: The residue of input after subtracting previous modes.
    """
    print('SVMD started.')
    print()
    start_time = timeit.default_timer()
    print('Input information:')
    assert y.ndim == 1, 'y.ndim = {y.ndim}'
    y_size = y.size
    print(f'y.size = {y_size}')
    if y_size % 2 != 0: # if ihe input size is odd,
        input_size_is_odd = True # no adjustment.
    else: # if ihe input size is even,
        y = np.append(y, 0.0)
        # then make it odd because even fft size will result in a frequency that is both positive and negative.
        input_size_is_odd = False # remember to use later when compensating for this adjustment.
        print('The input is padded 1 zero at the end because its size is even.')
    print(f'input_size_is_odd = {input_size_is_odd}')
    print()

    print('Decomposition information:')
    z = 2*scipy.fft.rfft(y, axis=0, norm='backward') # transform input to frequency domain. z represents complex.
    print(f'z.size = {z.size}')
    z_idx = np.arange(z.size)
    z_modes = []
    
    for k in range(1, out_iter_max+1):
        #mode_prev = np.amax(np.abs(z))
        mode_prev = z[np.argmax(np.abs(z))] # I'm not sure if this line or the above line is correct.
        
        for i in range(1, in_iter_max+1):
            mode_prev_sq = abs2(mode_prev)
            fcenter = np.sum(z_idx*mode_prev_sq)/np.sum(mode_prev_sq)
            z_prev = z - mode_prev
            z_prev_sq = abs2(z_prev)
            fcenter_res = np.sum(z_idx*z_prev_sq)/np.sum(z_prev_sq)
            mode_next = (z*(1 + beta*np.square(z_idx-fcenter_res)))/ \
                        (1+alpha*np.square(z_idx-fcenter) + beta*np.square(z_idx-fcenter_res))
            #if np.sum(np.square(np.abs(mode_next)-np.abs(mode_prev))) > in_thr:
            if np.sum(abs2(mode_next-mode_prev)) > in_thr: # I'm not sure if this line or the above line is correct.
                mode_prev = mode_next.copy()
            else:
                break

        print(f'The {k}th outer iteration took {i} inner iterations.')
        z_modes.append(mode_next)
        z -= mode_next
        if np.sum(abs2(z)) <= out_thr:
            break
        
    print(f'Totally {k+1} modes decomposed.')
    z_modes.append(z)
    z_modes = np.append(np.array(z_modes), np.zeros((k+1, y_size//2)), axis=1)
    y_modes = np.real(scipy.fft.ifft(z_modes, axis=1, norm='backward')) # transform output back to time domain.
    if not input_size_is_odd: # if input size is even,
        y_modes = np.delete(y_modes, -1, axis=1) # delete the last element of output to compensate.
    print('The last element of output is deleted because input size is even.')
    assert y_modes.shape[1] == y_size, f'y_modes.shape[1] = {y_modes.shape[1]}'
    print()
    end_time = timeit.default_timer()
    print(f'SVMD completed, running time: {round((end_time-start_time), 4)} seconds.')
    print()
    if return_type == 'modes':
        return y_modes
    elif return_type == 'modes, residual':
        return y_modes[:-1, :], y_modes[-1, :]
    else:
        raise ValueError(f'return_type "{return_type}" is not supported.')

def svmd_refine():
    print('Refined SVMD started.')
    start_time = timeit.default_timer()


    
    end_time = timeit.default_timer()
    print(f'Refined SVMD completed, running time: {round((end_time-start_time), 4)} seconds.')

    
