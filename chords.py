import util as UM
import mido
import musicnoteconv as mnc
import polars as pl

num_offsets = 12
num_inversions = 3
num_notes = 3
midi_upper_limit = mnc.note_to_midi('fs4')
offsets = {0: 'c4', 1: 'cs4', 2: 'd4', 3: 'ds4', 4: 'e4',
          5: 'f4', 6: 'fs4', 7: 'g4', 8: 'gs4', 9: 'a4',
          10: 'as4', 11: 'b4'}
chord_notes = {'major': ['c4', 'e4', 'g4'],
          'minor': ['c4', 'ef4', 'g4'],
          'dim': ['c4', 'ef4', 'gf4'],
          'aug': ['c4', 'e4', 'gs4']}
 

