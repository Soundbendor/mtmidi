import util as um
import mido
import musicnoteconv as mnc
import polars as pl


#inst is short instrument name
fieldnames = ['name', 'root', 'pitch', 'octave', 'quality', 'inversion', 'inst', 'quality_idx', 'rvb_lvl', 'rvb_val', 'bpm']
# including 0th offset
num_offsets = 12
num_inversions = 4
num_notes = 4
midi_upper_limit = mnc.note_to_midi('fs4')
offsets = {0: 'c4', 1: 'cs4', 2: 'd4', 3: 'ds4', 4: 'e4',
          5: 'f4', 6: 'fs4', 7: 'g4', 8: 'gs4', 9: 'a4',
          10: 'as4', 11: 'b4'}
chord_notes = {'major7': ['c4', 'e4', 'g4', 'b4'],
          'minor7': ['c4', 'ef4', 'g4', 'bf4'],
          'majorminor7': ['c4', 'e4', 'g4', 'bf4'],
          'minormajor7': ['c4', 'ef4', 'g4', 'b4'],
          'halfdim7': ['c4', 'ef4', 'gf4', 'bf4'],
          'fulldim7': ['c4', 'ef4', 'gf4', 'a4'],
          'augmajor7': ['c4', 'e4', 'gs4', 'b4'],
          'augminor7': ['c4', 'e4', 'gs4', 'bf4'],
          }

chord_midi = {k:[mnc.note_to_midi(y) for y in v] for (k,v) in chord_notes.items()}

chord_quality_to_idx = {x:i for (i,x) in enumerate(chord_notes.keys())}
idx_to_chord_quality = {i:x for (x,i) in chord_quality_to_idx.items()}

def make_inversion(cur_midinotes, inv):
    if inv == 0:
        return cur_midinotes
    else:
        notes = [x for x in cur_midinotes[inv:]]
        notes2 = [x + 12 for x in cur_midinotes[:inv]]
        ret = notes + notes2
        return ret

def offset_notes(cur_midinotes, offset):
    if offset == 0:
        return cur_midinotes
    else:
        return [x + offset for x in cur_midinotes]

def tranpose_to_range(cur_midinotes):
    transposed_down = False
    if cur_midinotes[0] >= midi_upper_limit:
        transposed_down = True
        return offset_notes(cur_midinotes, -12)
    else:
        return cur_midinotes, transposed_down

def get_outname(chord_type, inv_idx, short_inst, cur_root, rvb_lvl, ext = ""):
    ret = None
    outname = f'{chord_type}_inv{inv_idx}-{short_inst}-{cur_root}-rvb{rvb_lvl}'
    if len(ext) > 0:
        ret = f'{outname}.{ext}'
    else:
        ret = outname
    return ret

