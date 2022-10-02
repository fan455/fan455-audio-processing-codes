
def get_midi(note_idx, octave_idx):
    """
    For pianos' 88 notes only. There're 9 octaves (2 incomplete at both ends), counting from 0.
    """
    midi = 12 + note_idx + 12*octave_idx
    if 21 <= midi <= 108:
        return midi
    else:
        return 0

def get_midi_idx(midi):
    """
    For pianos' 88 notes only.
    """
    a, b = divmod(midi-11, 12)
    return b-1, a # note_idx, octave_idx

def get_midi_frequency(midi):
    f = 440*np.exp2((midi-69)/12)
    return f
