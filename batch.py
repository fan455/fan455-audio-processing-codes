"""
Audio batch processing template.
"""
import os
import soundfile as sf
from xxx import xxx #xxx is the function used for audio files batch processing.

in_f = '01'
out_f = '02'
os.makedirs(out_f)

k = 0
for root, dirs, files in os.walk(in_f):
    for file in files:
        if file.endswith('.wav'):
            in_p = os.path.join(root, file) 
            out_p = in_p.replace(in_f, out_f)
            y, sr = sf.read(in_p)
            y = xxx(y)
            sf.write(out_p, y, sr, subtype='PCM_24')
            k += 1
            print(f'{k} files processed.')

print('Batch processing completed.')
