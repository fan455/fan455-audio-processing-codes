import numpy as np
import soundfile as sf
import time_frequency_analysis as tfa

def get_sine_wave(f=440, phase=0, sr=48000, du=1):
    t = np.arange(0, int(sr*du))/sr
    return np.sin(2*np.pi*f*t + phase)

au, sr = get_sine_wave(phase=np.pi/2), 48000
#au, sr = sf.read('./audio_input/01.wav')

f, t, m = tfa.stft(au, sr, 1)
#f, t, z = tfa.stft(au, sr, 0, 'complex')
#f, t, r, i = tfa.stft(au, sr, 0, 'r, i')

tfa.plot_stft_m(f, t, m, frame=0)
#tfa.plot_stft_p(f, t, p, frame=0)
#tfa.plot_stft_z(f, t, z, frame=0)
#tfa.plot_stft_r(f, t, r, frame=0)
#tfa.plot_stft_i(f, t, i, frame=0)
