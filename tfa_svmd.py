# Sequential Variational Mode Decomposition (SVMD)
# This algorithm is developed by Wei Chen from JiangXi University of Finances and Economics.
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
from scipy import fft

def svmd(y, out_thr=1e-5, in_thr=1e-10, out_itr_max=20, in_itr_max=50, alpha=5e+1, beta=1e-1):
    """
    Parameters:
    y: 1d real array. The input signal array, need to be 1d, real, and better within range [-1, 1].
    out_thr: positive float. The threshold for outer iterations. A smaller value may result in more modes decomposed.
    in_thr: positive float. The threshold for inner iterations. A smaller value may result in more accurate modes decomposed.
    out_itr_max: positive int. Maxinum outer iteration times. It can avoid endless iteration case.
    in_itr_max: positive int. Maxinum inner iteration times. It can avoid endless iteration case.
    alpha: positive float. Penalty coffecient for the second quadratic term in the optimization.
    beta: positive float. Penalty coffecient for the third quadratic term in the optimization.

    Returns:
    y_Mode: nd real array. The decomposed modes of y of shape (number of modes, size of y).
        The last mode is the residue of input after subtracting previous modes.
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
    z = 2*fft.rfft(y, axis=0, norm='forward') # transform input to frequency domain. z represents complex.
    print(f'z.size = {z.size}')
    z_idx = np.arange(z.size)
    z_res = z.copy()
    z_Mode = []
    
    for k in range(1, out_itr_max+1):
        mode_prev = np.amax(np.abs(z_res))
        
        for i in range(1, in_itr_max+1):
            mode_prev_square = np.square(np.abs(mode_prev))
            fcenter = np.sum(z_idx*mode_prev_square)/np.sum(mode_prev_square)
            z_res_prev = z_res - mode_prev
            z_res_prev_square = np.square(np.abs(z_res_prev))
            fcenter_res = np.sum(z_idx*z_res_prev_square)/np.sum(z_res_prev_square)
            mode_next = (z_res*(1 + beta*np.square(z_idx-fcenter_res)))/ \
                        (1+alpha*np.square(z_idx-fcenter) + beta*np.square(z_idx-fcenter_res))
            if np.sum(np.square(np.abs(mode_next)-np.abs(mode_prev))) > in_thr:
                mode_prev = mode_next.copy()
            else:
                break

        print(f'The {k}th outer iteration took {i} inner iterations.')
        z_Mode.append(mode_next)
        z_res -= mode_next
        if np.sum(np.square(np.abs(z_res))) > out_thr:
            pass
        else:
            break
        
    print(f'Totally {k+1} modes decomposed.')
    z_Mode.append(z_res)
    z_Mode = np.array(z_Mode)
    z_Mode = np.append(z_Mode, np.zeros((k+1, y_size//2)), axis=1)
    y_Mode = np.real(fft.ifft(z_Mode, axis=1, norm='forward')) # transform output back to time domain.
    if not input_size_is_odd: # if input size is even,
        y_Mode = np.delete(y_Mode, -1, axis=1) # delete the last element of output to compensate.
    print('The last element of output is deleted because input size is even.')
    assert y_Mode.shape[1] == y_size, f'y_Mode.shape[1] = {y_Mode.shape[1]}'
    print()
    end_time = timeit.default_timer()
    print(f'SVMD completed, running time: {round((end_time-start_time), 4)} seconds.')
    return y_Mode
        

