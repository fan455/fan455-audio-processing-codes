import numpy as np
import soundfile as sf
import tfa
from pitch import note2freq
"""
The test audio 'A6v8.wav' is derived from the bitklavier grand piano sample library.
"""
def get_sine_wave(f=440, phase=0, sr=48000, du=1):
    t = np.arange(0, int(sr*du))/sr
    return np.sin(2*np.pi*f*t + phase)

au, sr = sf.read('audio_input/A6v8.wav')

given_freq = note2freq('A6')
given_cent = 50
pitch = tfa.get_pitch_given(au, sr, channel=0, given_freq=given_freq, given_cent=given_cent)

