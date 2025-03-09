from datasets import load_dataset
import csv
import util as um
import os


# format
#{'audio': {'path': 'C_major_5_1_Bright_Acoustic_Piano.wav', 'sampling_rate': 44100},
#'root_note_name': 'C', 'chord_type': 'major', 'inversion': 5, 'root_note_is_accidental': False, 'root_note_pitch_class': 0, 'midi_program_num': 1, 'midi_program_name': 'Bright Acoustic Piano', 'midi_category': 'Piano Timbres'}


out_path = um.by_projpath(subpath='hf_csv',make_dir = True)
out_f = os.path.join(out_path, 'chords.csv')
bool_keys = set(['root_note_is_accidental'])
ignore_keys = set(['array', 'audio'])
audio_keys = set(['path', 'array', 'sampling_rate'])
toplevel_keys = set(['audio','root_note_name', 'chord_type', 'inversion', 'root_note_is_accidental', 'root_note_pitch_class', 'midi_program_num', 'midi_program_name', 'midi_category'])

use_keys = [x for x in toplevel_keys.union(audio_keys) if x not in ignore_keys]
use_keys += ['name']

chords = load_dataset("meganwei/syntheory", "chords", split='train')

with open(out_f, 'w') as f:
    csvw = csv.DictWriter(f, fieldnames=use_keys)
    csvw.writeheader()

    for x in chords:
        d = {}
        for k1 in toplevel_keys:
            if k1 not in ignore_keys:
                if k1 not in bool_keys:
                    d[k1] = x[k1]
                else:
                    if x[k1] == True:
                        d[k1] = 1
                    else:
                        d[k1] = 0
        for k2 in x['audio'].keys():
            if k2 not in ignore_keys:
                d[k2] = x['audio'][k2]
        d['name'] = um.ext_replace(x['audio']['path'], new_ext="")
        csvw.writerow(d)

