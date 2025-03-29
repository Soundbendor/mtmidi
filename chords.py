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
chord_notes = {'major': ['c4', 'e4', 'g4'],
          'minor': ['c4', 'ef4', 'g4'],
          'dim': ['c4', 'ef4', 'gf4'],
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

def transpose_to_range(cur_midinotes):
    transposed_down = False
    if cur_midinotes[0] >= midi_upper_limit:
        transposed_down = True
        return offset_notes(cur_midinotes, -12), transposed_down
    else:
        return cur_midinotes, transposed_down


