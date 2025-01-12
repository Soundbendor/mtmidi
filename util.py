# delegating all random stuff here since I'm setting the seed here

import sys
import mido
import csv
import os
from midi2audio import FluidSynth
#import fluidsynth as fl
import numpy as np
import random
import librosa
import soundfile as sf
import tomllib
#import pyaudio
#import scipy.io.wavfile as siw
seed = 5
inst = {}
drum = {}

random.seed(5)
bpms = (1./60000.) # (1 min/60 sec) x (1 sec/1000 ms)

drum_pfix = "D:"
drum_pgm = 0
drum_chnum = 9
default_midinote = 60

# datasets in the original dataset (for debugging)
hf_datasets = set(['tempos', 'time_signatures', 'chords', 'notes', 'scales', 'intervals', 'chord_progressions']) 

# datasets this project is introducing
new_datasets = set(['polyrhythms'])

all_datasets = hf_datasets.union(new_datasets)

model_longhand = {'mg_audio': 'musicgen-encoder',
                 'mg_small_h': 'musicgen-small_hidden', 'mg_med_h': 'musicgen-medium_hidden', 'mg_large_h': 'musicgen-large_hidden',
                 'mg_small_at': 'musicgen-small_attn', 'mg_med_at': 'musicgen-medium_attn', 'mg_large_at': 'musicgen-large_attn',
                  'jukebox': 'jukebox', 'jukebox36': 'jukebox36', 'jukebox38': 'jukebox38'}

model_shorthand = model_longhand.keys()
model_type = {'musicgen-encoder': 'musicgen-large',
              'musicgen-small_hidden': 'musicgen-small',
              'musicgen-small_attn': 'musicgen-small',
              'musicgen-medium_hidden': 'musicgen-medium',
              'musicgen-medium_attn': 'musicgen-medium',
              'musicgen-large_hidden': 'musicgen-large',
              'musicgen-large_attn': 'musicgen-large',
              'jukebox': 'jukebox',
              'jukebox36': 'jukebox',
              'jukebox38': 'jukebox'}

model_sr = {'jukebox': 44100, 'musicgen-encoder': 32000,
            'musicgen-small': 32000, 'musicgen-medium': 32000,
            'musicgen-large': 32000}

# musicgen-medium/large: 48 + 1
# musicgen-small: 24 + 1
model_num_layers = {"musicgen-small": 25, "musicgen-medium": 49, "musicgen-large": 49, "musicgen-encoder": 1, "jukebox": 72}
#model_num_layers = {"musicgen-small": 24, "musicgen-medium": 48, "musicgen-large": 48, "musicgen-encoder": 1, "jukebox": 1, "jukebox36": 1, "jukebox38": "jukebox38"} #until we get all jukebox layers

# because jukebox does a weird 1-indexing thing
jukebox_range = list(range(1,73))

act_layer_dim = {"musicgen-small_hidden": 1024, "musicgen-medium_hidden": 1536, "musicgen-large_hidden": 2048, 
                   "musicgen-small_attn": 16, "musicgen-medium_attn": 24, "musicgen-large_attn": 32,
                   "musicgen-encoder": 128,
                 "jukebox": 4800,
                 "jukebox36": 4800,
                 "jukebox38": 4800,
                 }

act_folder = {'musicgen-encoder': 'mg_audio_mp',
              'musicgen-small_hidden': 'mg_small_mp',
              'musicgen-small_attn': 'mg_small_mp',
              'musicgen-medium_hidden': 'mg_medium_mp',
              'musicgen-medium_attn': 'mg_medium_mp',
              'musicgen-large_hidden': 'mg_large_mp',
              'musicgen-large_attn': 'mg_large_mp',
              'jukebox': 'jukebox_acts_36',
              'jukebox36': 'jukebox_acts_36',
              'jukebox38': 'jukebox_acts_38',
              }

# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

def by_projpath(subpath=None,make_dir = False):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    if subpath != None:
        cur_path = os.path.join(cur_path, subpath)
        if os.path.exists(cur_path) == False and make_dir == True:
            os.makedirs(cur_path)
    return cur_path

def get_model_type(shorthand):
    return model_type[model_longhand[shorthand]]

def get_embedding_shape(shorthand):
    longhand = model_longhand[shorthand]
    mtype = model_type[longhand]
    num_layers = model_num_layers[mtype]
    layer_dim = act_layer_dim[longhand]
    shape = (num_layers, layer_dim)
    return shape

def get_embedding_file(model_shorthand, acts_folder = 'acts', model_folder = 'jukebox', dataset='polyrhythms', fname='', write = True, use_64bit = True):
    actpath = by_projpath(acts_folder)
    datapath = os.path.join(actpath, dataset)
    modelpath = os.path.join(datapath, model_folder)
    fpath = os.path.join(modelpath, fname)
    fp = None
    dtype = 'float32'
    mode = 'r'
    shape = get_embedding_shape(model_shorthand)
    if use_64bit == True:
        dtype = 'float64'
    if write == True:
        if os.path.exists(datapath) == False:
            os.makedirs(datapath)
        if os.path.exists(modelpath) == False:
            os.makedirs(modelpath)



        if os.path.isfile(fpath) == True:
            mode = 'r+'
        else:
            mode = 'w+'
    fp = np.memmap(fpath, dtype = dtype, mode=mode, order='C', shape=shape)
    return fp

with open(by_projpath('inst_list.csv'), 'r') as f:
    csvr = csv.reader(f, delimiter=',')
    for i,row in enumerate(csvr):
        if i > 0:
            inst_num = int(row[0])
            inst_name = row[1]
            inst_cat = row[2]
            inst[inst_name] = {'program_number': inst_num, 'category': inst_cat}

with open(by_projpath('drum_list.csv'), 'r')as f:
    csvr = csv.reader(f, delimiter=",")
    for i,row in enumerate(csvr):
        if i > 0:
            prg_num = int(row[0])
            note_num = int(row[1])
            inst_name = row[2].strip()
            gm2 = True if int(row[3])== 1 else 0
            drum[inst_name] = {'program_number': prg_num, 'midinote': note_num, 'gm2': gm2}


def shuf_arr(arr):
    random.shuffle(arr)

def read_toml(sysargv, toml_dir = os.path.join(script_dir, 'toml')):
    fname = 'default.toml'
    if len(sysargv) > 1:
        fname = sysargv[1]
    fpath = os.path.join(toml_dir, fname)
    toml_file = None
    try:
        with open(fpath, 'rb') as f:
            toml_file = tomllib.load(f)
            print(f'read {fpath}')
        return toml_file
    except:
        print(f'error reading {fpath}')
        quit()

def path_list(subpath=None):
    cur_path = by_projpath(subpath=subpath, make_dir = False)
    return os.listdir(cur_path) 

def get_random_list(lo, hi, num):
    return [random.uniform(lo,hi) for _ in range(num)]

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


def get_drum_name(cur_inst):
    dsplit = cur_inst.strip().split(drum_pfix[-1])
    return dsplit[-1]

def is_inst_drum(cur_inst):
    if drum_pfix in cur_inst:
        return True
    else:
        return False

def get_inst_program_number(cur_inst):
    if drum_pfix in cur_inst:
        return drum_pgm
    else:
        return inst[cur_inst]['program_number']

def get_inst_midinote(cur_inst, default=default_midinote):
    ret = default
    if drum_pfix in cur_inst:
        try:
            drum_name = get_drum_name(cur_inst)
            ret = drum[drum_name]['midinote']
        except:
            ret = default
    return ret


def save_midi(midifile, midiname, save_dir = "midi", dataset='polyrhythms'):
    sub_dir = os.path.join(by_projpath(save_dir), dataset)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    if os.path.exists(sub_dir) == False:
        os.makedirs(sub_dir)
    fullpath = os.path.join(sub_dir, midiname)
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

    
def ms_to_ticks(ms, ticks_per_beat = 1000, bpm = 120):
    # ticks/beat x beats/min x min/sec x sec/ms = ticks/ms
    # ticks/beat x beats/ms = ticks/ms
    # ticks/ms x ms = ticks
    ticks = (ticks_per_beat * bpm) * bpms
    return ticks

# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
# takes mean of stereo channels (doesn't rely on loading as mono)
# normalizes via numpy (divide by max)
def load_wav(fname, dur = 4., normalize = False, sr=32000, load_dir='wav'):
    wavpath = os.path.join(load_dir, fname)
    snd, load_sr = librosa.load(wavpath, duration = dur, mono = True, sr=sr)
    if normalize == False:
        return snd
    else:
        return librosa.util.normalize(snd)
    
# replace extension from path
def ext_replace(old_path, new_ext = 'pt'):
    fsplit = '.'.join(old_path.split('.')[:-1])
    outname = fsplit
    if len(new_ext) > 0:
        outname = f'{fsplit}.{new_ext}'
    return outname


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
    midipath = by_projpath('midi')
    return [os.path.join(midipath, x)  for x in os.listdir(midipath) if '.mid' in x]

def series_plot_save(pl_series, cat, fname='chart.png', ds='polyrhy_split1'):
    title = f'{ds} {cat} counts'
    ch = pl_series.plot.bar(x=cat, y='count', text='count')
    ch.title = title
    ax = ch.mark_bar() + ch.mark_text(dy=-6)
    ax.save(fname)

def profile_category(df, cat, ds='polyrhy_split1', profile_dir = 'dataprof', profile_type='overall', save_csv = True, save_png = True):
    if os.path.exists(profile_dir) == False:
        os.makedirs(profile_dir)
    dp_projdir = os.path.join(profile_dir, ds)

    if os.path.exists(dp_projdir) == False:
        os.makedirs(dp_projdir)
    dp_png_dir = os.path.join(dp_projdir, 'png')
    dp_csv_dir = os.path.join(dp_projdir, 'csv')
    out_name2 = f'{profile_type}_{cat}'
    ser = df[cat].value_counts()
    if save_png == True:
        if os.path.exists(dp_png_dir) == False:
            os.makedirs(dp_png_dir)

        opng = os.path.join(dp_png_dir, f'{out_name2}.png' )
        series_plot_save(ser, cat, fname=opng, ds=ds)

    if save_csv == True:
        if os.path.exists(dp_csv_dir) == False:
            os.makedirs(dp_csv_dir) 
        ocsv = os.path.join(dp_csv_dir, f'{out_name2}.csv')
        ser.write_csv(ocsv)



