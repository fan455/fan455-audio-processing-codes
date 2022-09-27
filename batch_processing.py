import os
from xxx import xxx
"""
xxx is the function used for audio files batch processing.
"""

input_folder = '01'
output_folder = '02'

k = 0
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.wav'):
            input_path = os.path.join(root, file) 
            output_path = input_path.replace(input_folder, output_folder)
            xxx.xxx(input_path, output_path)
            k += 1
            print(f'{k} files processed.')

print('Batch processing completed.')


