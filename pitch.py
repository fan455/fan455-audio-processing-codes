"""
Musical pitch calculations including note, midi, frequency and cent conversions.
For pianos' 88 notes only. There're 9 octaves (2 incomplete octaves at both ends), counting from 0.
"""
Note = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

def note2midi(note_str, middle_c='C4'):
    if len(note_str) == 2:
        note = note_str[0]
    elif len(note_str) == 3:
        note = note_str[0: 2]
    else:
        raise ValueError('The length of note_str needs to be 2 or 3')
    note = note.upper()
    note_idx = Note.index(note)
    octave_idx = int(note_str[-1]) + 4 - int(middle_c[-1])
    midi = idx2midi(note_idx, octave_idx)
    return midi

def midi2note(midi, middle_c='C4'):
    note_idx, octave_idx = midi2idx(midi)
    note = Note[note_idx]
    octave = octave_idx + int(middle_c[-1]) - 4
    return f'{note}{octave}'

def idx2midi(note_idx, octave_idx):
    midi = 12 + note_idx + 12*octave_idx
    if 21 <= midi <= 108:
        return midi
    else:
        return 0 # Only indices of the 88 notes will have true corresponding midi, otherwise returns 0.

def midi2idx(midi):
    a, b = divmod(midi-11, 12)
    return b-1, a # note_idx, octave_idx

def midi2freq(midi):
    f = 440*np.exp2((midi-69)/12)
    return f

def note2freq(note_str, middle_c='C4'):
    midi = note2midi(note_str, middle_c=middle_c)
    f = midi2freq(midi)
    return f

def 2f2cent(f1, f2):
    cent = 1200*np.log2(f2/f1)
    return cent

def 2midi2cent(midi1, midi2):
    f1, f2 = midi2freq(midi1), midi2freq(midi2)
    cent = 2f2cent(f1, f2)
    return cent

def ratio2cent(ratio):
    cent = 1200*np.log2(ratio)
    return cent

def cent2ratio(cent):
    ratio = np.exp2(cent/1200)
    return ratio
