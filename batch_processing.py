"""
Audio batch processing template.
"""
import os
import soundfile as sf
from xxx import xxx #xxx is the function used for audio files batch processing.

input_folder = '01'
output_folder = '02'

k = 0
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.wav'):
            input_path = os.path.join(root, file) 
            output_path = input_path.replace(input_folder, output_folder)
            #output_path = output_path.replace('?', 'wav')
            y, sr = sf.read(input_path)
            y = xxx(y)
            sf.write(output_path, y, sr, subtype='PCM_24')
            k += 1
            print(f'{k} files processed.')

print('Batch processing completed.')
