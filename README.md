# fan455-audio-processing-codes
The audio processing codes I wrote for python.

Required python packages: numpy, scipy, matplotlib, soundfile.

IMPORTANT: Please note that in this repository, the shapes of all more-than-1-channel audio arrays follow the Soundfile, i.e. if audio.ndim>=2, audio.shape=(number_of_samples, number_of_channels), while some libraries like librosa and resampy use the reverse shape.
