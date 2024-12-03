from datasets import load_dataset
import csv
import util as um
import os


# train is a key of tempi with value array

out_path = um.by_projpath(subpath='hf_csv',make_dir = True)
out_f = os.path.join(out_path, 'time_signatures.csv')
ignore_keys = set(['array', 'audio'])
audio_keys = set(['path', 'array', 'sampling_rate'])
# time_signature: tuple 
# time_signature_beats: num
# time_singature subdivision: denom
# is_compound: int 0-1 for compound meter
# bpm = always 120
# click_config_name: midi inst name
# midi_program_num: int
# offset_time: float
# reverb_level = int
toplevel_keys = set(['time_signature', 'time_signature_beats', 'time_signature_subdivision', 'is_compound', 'bpm', 'click_config_name', 'midi_program_num', 'offset_time', 'reverb_level'])

use_keys = [x for x in toplevel_keys.union(audio_keys) if x not in ignore_keys]
use_keys += ['name']

timesig = load_dataset("meganwei/syntheory", "time_signatures")
with open(out_f, 'w') as f:
    csvw = csv.DictWriter(f, fieldnames=use_keys)
    csvw.writeheader()

    for x in timesig['train']:
        d = {}
        for k1 in toplevel_keys:
            if k1 not in ignore_keys:
                d[k1] = x[k1]
        for k2 in x['audio'].keys():
            if k2 not in ignore_keys:
                d[k2] = x['audio'][k2]
        d['name'] = um.ext_replace(x['audio']['path'], new_ext="")
        csvw.writerow(d)
