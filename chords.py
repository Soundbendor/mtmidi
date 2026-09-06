import util as UM
import mido
import musicnoteconv as mnc
import polars as pl

num_offsets = 12
num_inversions = 3
num_notes = 3
c4midi = mnc.note_to_midi('c4')
midi_upper_limit = mnc.note_to_midi('fs4')
offsets = {0: 'c4', 1: 'cs4', 2: 'd4', 3: 'ds4', 4: 'e4',
          5: 'f4', 6: 'fs4', 7: 'g4', 8: 'gs4', 9: 'a4',
          10: 'as4', 11: 'b4'}
note_offsets = {y:x for (x,y) in offsets.items()}
chord_notes = {'maj': ['c4', 'e4', 'g4'],
          'min': ['c4', 'ds4', 'g4'],
          'dim': ['c4', 'ds4', 'fs4'],
          'aug': ['c4', 'e4', 'gs4']}
 
def get_distance_from_c4(notename):
    curmidi = mnc.note_to_midi(notename)
    return curmidi-c4midi

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

def transpose_to_range(cur_midinotes, inv_idx):
    transposed_down = False
    cur_root_idx = (4 - inv_idx) % 4
    cur_root = mnc.midi_to_note(cur_midinotes[cur_root_idx], sharp = True)
    if cur_midinotes[cur_root_idx] >= midi_upper_limit:
            transposed_down = True
            while cur_midinotes[cur_root_idx] >= midi_upper_limit:
                cur_midinotes = offset_notes(cur_midinotes, -12)
            cur_root = mnc.midi_to_note(cur_midinotes[cur_root_idx], sharp = True)
    return cur_midinotes, transposed_down, cur_root

