import numpy as np
import soundfile as sf

def separate_channels(input_name, input_folder, output_folder, sf_subtype='PCM_16'):
    input_path = input_folder + '/' + input_name
    au, sr = sf.read(input_path)
    channels_num = au.shape[-1]
    for i in range(0, channels_num):
        au_sep = au[:, i]
        output_path = output_folder + '/' + input_name.replace('.', f'c{i+1}.')
        sf.write(output_path, au_sep, sr, sf_subtype)
        print(f"Channel {i+1} separated. Written to '{output_path}'.")
