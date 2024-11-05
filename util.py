import mido
import csv
import os
from midi2audio import FluidSynth
#import fluidsynth as fl
import numpy as np
import random
import librosa
import soundfile as sf
#import pyaudio
#import scipy.io.wavfile as siw
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

def write_to_wav_pyfl(midifilepath, save_dir = 'wav', sr = 44100, gain=0.2, channels=16, sec=4):
    syn = fl.Synth(gain=gain, samplerate=sr, channels=channels)
    mfpsplit = os.path.basename(midifilepath).split('.')
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    outfilepath = os.path.join(save_dir, '.'.join(mfpsplit[:-1]) + '.wav')
    # snippet from https://github.com/nwhitehead/pyfluidsynth/blob/21e30cc9e245b4a28b7d39a60932dcf5be582461/test/test6.py
    sfid = syn.sfload(os.path.join(os.path.dirname(__file__), 'TimGM6mb.sf2'))
    syn.program_select(0, sfid, 0, 0)
    syn.custom_router_callback = None
    syn.play_midi_file(midifilepath)
    s = []
    for _ in range(sec):
        s = np.append(s, syn.get_samples(sr))
        if fl.fluid_player_get_status(syn.player) != fl.FLUID_PLAYER_PLAYING:
            break
    syn.delete()
    siw.write(outfilepath, sr, s)

    


def write_to_wav(midifilepath, sr = 44100, save_dir = "wav"):
    fs = FluidSynth(sample_rate=sr, sound_font = 'TimGM6mb.sf2')
    mfpsplit = os.path.basename(midifilepath).split('.')
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    outfilepath = os.path.join(save_dir, '.'.join(mfpsplit[:-1]) + '.wav')
    #print(outfilepath)
    fs.midi_to_audio(midifilepath, outfilepath)

def clean_wav(wavpath, out_dir,sr=44100):
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    wavpath = os.path.join(wav_dir, x)
    snd, load_sr = librosa.load(wavpath, mono = True, sr=sr)
    snd_trim = snd[:want_samp]
    wsplit = x.split('.')
    wjoin = '.'.join(wsplit[:-1]) + '-trim.wav'
    opath = os.path.join(our_dir, wjoin)
    sf.write(opath, snd_trim, sr)

def get_saved_midi():
    return [os.path.join('midi', x)  for x in os.listdir('midi') if '.mid' in x]
