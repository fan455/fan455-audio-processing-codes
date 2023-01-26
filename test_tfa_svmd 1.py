# SVMD test using signal 1 from the paper 'A Sequential Variational Mode Decomposition Method'.

import numpy as np
from plot import plot, plot_modes_residual
from tfa_svmd import *
from tfa import get_rfftm

# Define signal
sr, du, du_extend = 5000, 1.0, 0.2
y = np.load('signal 1 extended.npy')
t = np.arange(int(sr*(du+2*du_extend)))/sr - du_extend

#y = 2*t + np.sin(100*np.pi*t-10*np.pi*np.square(t)) + 0.5*np.exp(-5*np.square(t-0.5))* \
     #np.sin(200*np.pi*t) + np.random.normal(scale=0.1, size=t.size)
#np.save('signal 1 extended.npy', y)

# Set parameters
out_thr, in_thr = 1e-5, 1e-10
out_iter_max, in_iter_max = 3, 150
alpha, beta = 1, 1e-2
merge_range = 1.5
return_type = 'modes, residual'

# Decompose
y_rfft = get_rfftm(y)
Modes, res = svmd(y, out_thr, in_thr, out_iter_max, in_iter_max, alpha, beta, return_type)
#Modes, res = svmd_refined(y, out_thr, in_thr, out_iter_max, in_iter_max, alpha, beta, merge_range, return_type)
res_rfft = get_rfftm(res)

if Modes.shape[0] <= 10:
    pass
elif 10 < Modes.shape[0] <= 20:
    #Modes = Modes[:10, :]
    Modes = Modes[10:, :]
elif 20 < Modes.shape[0] <= 30:
    #Modes = Modes[:10, :]
    #Modes = Modes[10:20, :]
    Modes = Modes[20:, :]
elif 30 < Modes.shape[0] <= 40:
    #Modes = Modes[:10, :]
    #Modes = Modes[10:20, :]
    #Modes = Modes[20:30, :]
    Modes = Modes[30:, :]
elif 40 < Modes.shape[0] <= 50:
    #Modes = Modes[:10, :]
    #Modes = Modes[10:20, :]
    #Modes = Modes[20:30, :]
    #Modes = Modes[30:40, :]
    Modes = Modes[40:, :]
elif 50 < Modes.shape[0] <= 60:
    #Modes = Modes[:10, :]
    #Modes = Modes[10:20, :]
    #Modes = Modes[20:30, :]
    #Modes = Modes[30:40, :]
    #Modes = Modes[40:50, :]
    Modes = Modes[50:, :]
else:
    raise ValueError('Too many modes.')

# Plot
plot(y_rfft, title='spectrum of the original signal', x_label='frequency', y_label='magnitude')
plot(res_rfft, title='spectrum of the residual', x_label='frequency', y_label='magnitude')
plot_modes_residual(Modes, res, y, t)
