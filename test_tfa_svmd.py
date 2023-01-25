import numpy as np
from plot import *
from tfa_svmd import *
from tfa import rfft_class

# Define signal
sr = 5000
t = np.arange(sr)/sr
y = np.load('signal 1.npy')
#y = 2*t + np.sin(100*np.pi*t-10*np.pi*np.square(t)) + 0.5*np.exp(-5*np.square(t-0.5))* \
     #np.sin(200*np.pi*t) + np.random.normal(scale=0.1, size=t.size)
#np.save('signal 1.npy', y)

# Decompose
class1 = rfft_class(fft_type='m')
y_rfft = class1.fw(y)
plot(y_rfft, title='spectrum of the original signal', x_label='frequency', y_label='magnitude')

Modes, res = svmd(y, out_thr=1e-5, in_thr=1e-5, out_iter_max=3, in_iter_max=50, alpha=1, beta=5, return_type='modes, residual')
res_rfft = class1.fw(res)
plot(res_rfft, title='spectrum of the residual', x_label='frequency', y_label='magnitude')

if Modes.shape[0] <= 10:
    pass
elif 10 < Modes.shape[0] <= 20:
    #Modes = Modes[:10, :]
    Modes = Modes[10:, :]
elif 20 < Modes.shape[0] <= 30:
    #Modes = Modes[:10, :]
    Modes = Modes[10:20, :]
    #Modes = Modes[20:, :]
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
#plot_modes(Modes, y, sr)
plot_modes_residual(Modes, res, y, sr)
