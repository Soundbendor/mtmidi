import mido
import csv
import os
from midi2audio import FluidSynth

import random

seed = 5
inst = {}

random.seed(5)

with open('inst_list.csv', 'r') as f:
    csvr = csv.reader(f, delimiter=',')
    for i,row in enumerate(csvr):
        if i > 0:
            inst_num = int(row[0])
            inst_name = row[1]
            inst_cat = row[2]
            inst[inst_name] = {'number': inst_num, 'category': inst_cat}

def shuffle_list(cur_list):
    random.shuffle(cur_list)

def coinflip_label(chance= 0.5, label1 = 'val', label2='test'):
    cur_rand = random.random()
    ret = label1
    if cur_rand >= chance:
        ret = label2
    return ret

def notedur_to_ticks(dur, subdiv = 1, ticks_per_beat = 1000, sustain=1.0):
    on_dur = ticks_per_beat
    if dur != 4:
        on_dur *= 4./dur
    if subdiv != 1:
        on_dur /= subdiv
    off_dur = 0
    if sustain < 1.:
        orig = on_dur
        on_dur = int(min(sustain * orig, orig))
        off_dur = int(max(orig - on_dur, 0))
    else:
        on_dur = int(on_dur)
    return (on_dur, off_dur)

def get_inst_program_number(cur_inst):
    return inst[cur_inst]['number']

def save_midi(midifile, midiname, save_dir = "midi"):
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    fullpath = os.path.join(save_dir, midiname)
    midifile.save(fullpath)

def write_to_wav(midifilepath, sr = 44100, save_dir = "wav"):
    fs = FluidSynth(sample_rate=sr, sound_font = 'TimGM6mb.sf2')
    mfpsplit = os.path.basename(midifilepath).split('.')
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    outfilepath = os.path.join(save_dir, '.'.join(mfpsplit[:-1]) + '.wav')
    #print(outfilepath)
    fs.midi_to_audio(midifilepath, outfilepath)

def get_saved_midi():
    return [os.path.join('midi', x)  for x in os.listdir('midi') if '.mid' in x]
