from datasets import load_dataset
import csv
import util as um
import os


# train is a key of tempi with value array
# each element of tempi['train'] has keys (['audio', 'bpm', 'click_config_name', 'midi_program_num', 'offset_time'])
# audio is keyed (['path', 'array', 'sampling_rate'])
# audio.path is filename
#audio.array is array of audio samples?
# audio.sampling_rate is sampling rate
# bpm is int
# click_config_name is name of instrument as str
# midi_program_num is int
# offset_time is float in seconds?

out_path = um.by_projpath(subpath='hf_csv',make_dir = True)
out_f = os.path.join(out_path, 'tempi.csv')
ignore_keys = set(['array', 'audio'])
audio_keys = set(['path', 'array', 'sampling_rate'])
toplevel_keys = set(['audio', 'bpm', 'click_config_name', 'midi_program_num', 'offset_time'])

use_keys = [x for x in toplevel_keys.union(audio_keys) if x not in ignore_keys]
use_keys += ['norm_bpm', 'name']
minbpm = 50
maxbpm = 210
bpmrange = maxbpm - minbpm
bpmscaler = lambda x: float(x - minbpm)/bpmrange

tempi = load_dataset("meganwei/syntheory", "tempos", split='train')
with open(out_f, 'w') as f:
    csvw = csv.DictWriter(f, fieldnames=use_keys)
    csvw.writeheader()

    for x in tempi:
        d = {}
        for k1 in toplevel_keys:
            if k1 not in ignore_keys:
                d[k1] = x[k1]
        for k2 in x['audio'].keys():
            if k2 not in ignore_keys:
                d[k2] = x['audio'][k2]
        d['norm_bpm'] = bpmscaler(x['bpm'])
        d['name'] = um.ext_replace(x['audio']['path'], new_ext="")
        csvw.writerow(d)



"""
for x in tempi['train']:
    print(x)
#print(len(tempi['train']))
for i,x in enumerate(tempi['train']):
    cur = x['audio']
    print(cur['path'], cur['sampling_rate'])
    print(i,cur['array'].shape)
"""
#print(tempi['train'][30])
