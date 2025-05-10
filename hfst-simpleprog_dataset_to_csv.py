from datasets import load_dataset
import csv
import util as um
import os


# train is a key of tempi with value array

out_path = um.by_projpath(subpath='hf_csv',make_dir = True)
out_f = os.path.join(out_path, 'simple_progressions.csv')
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


def parse_chord_progression_str(chpr_str):
    is_major = False
    dashsp = chpr_str.split('-')
    cur_mode = dashsp[0].strip()
    if cur_mode == 'ionian':
        is_major = True
    nums = [x.strip() for x in dashsp[1].strip().split(',')]
    num_nums = len(nums)
    prog = []
    for i,x in enumerate(nums):
        if i == 0:
            prog.append(int(x[1:]))
        elif i == (num_nums - 1):
            prog.append(int(x[:1]))
        else:
            prog.append(int(x))
    prog_str = ''.join([str(x) for x in prog])
    strpfx = 'maj'
    if is_major == False:
        strpfx = 'min'
    ret_str = f'{strpfx}-{prog_str}'
    return (is_major, ret_str)




# chord_progression example:
# ionian-(1, 4, 5, 1)
# aeolian-(7, 6, 7, 1)
toplevel_keys = set(['key_note_name', 'key_note_pitch_class', 'chord_progression', 'midi_program_num', 'midi_program_name', 'midi_category'])
#toplevel_keys = set(['time_signature', 'time_signature_beats', 'time_signature_subdivision', 'is_compound', 'bpm', 'click_config_name', 'midi_program_num', 'offset_time', 'reverb_level'])

use_keys = [x for x in toplevel_keys.union(audio_keys) if x not in ignore_keys]
use_keys += ['name']
use_keys += ['is_major', 'orig_prog', 'inv', 'bpm']
simpleprog = load_dataset("meganwei/syntheory", "simple_progressions")
inv = 0
bpm = 60
with open(out_f, 'w') as f:
    csvw = csv.DictWriter(f, fieldnames=use_keys)
    csvw.writeheader()

    for x in simpleprog['train']:
        d = {}
        for k1 in toplevel_keys:
            if k1 not in ignore_keys:
                d[k1] = x[k1]
        for k2 in x['audio'].keys():
            if k2 not in ignore_keys:
                d[k2] = x['audio'][k2]
        d['inv'] = inv
        d['bpm'] = bpm
        d['name'] = um.ext_replace(x['audio']['path'], new_ext="")
        cur_ismajor, cur_progstr = parse_chord_progression_str(d['chord_progression'])
        d['is_major'] = cur_ismajor
        d['orig_prog'] = cur_progstr
        csvw.writerow(d)
