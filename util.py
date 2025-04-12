# delegating all random stuff here since I'm setting the seed here

import torch
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

reverb_lvl = {0:0, 1: 63, 2:127}
# datasets in the original dataset (for debugging)
hf_datasets = set(['tempos', 'time_signatures', 'chords', 'notes', 'scales', 'intervals', 'simple_progressions'])

# datasets this project is introducing
new_datasets = set(['polyrhythms', 'dynamics', 'chords7', 'secondary_dominant', 'modemix_chordprog'])


#datasets that are regression
reg_datasets = set(['tempos'])
# datasets to train on middle on
tom_datasets = set(['tempos'])
baseline_names = set(['baseline_mel', 'baseline_chroma', 'baseline_mfcc', 'baseline_concat'])
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

# decoder.hidden_states
# --- large: 1, 200, 2048 (49 hidden states, 0-47: AddBackward0, 48: NativeLayerNormBackward0)
# --- medium: 1, 200, 1536 (49 hidden states, 0-47: AddBackward0, 48: NativeLayerNormBackward0)
# --- small: 1, 200, 1024 (25 hidden states, 0-47: AddBackward0, 48: NativeLayerNormBackward0)

# decoder.attention
# --- large: 1, 32, 200, 200 (48, all ViewBackward0)
# --- medium: 1, 24, 200, 200 (48, all ViewBackward0)
# --- medium: 1, 16, 200, 200 (24, all ViewBackward0)

# from https://huggingface.co/docs/transformers/main/en/model_doc/musicgen#transformers.MusicgenForConditionalGeneration:
# output of each hidden layer of decoder plus initial embedding outputs (so before decoder)
# musicgen-medium/large: 48 + 1
# musicgen-small: 24 + 1
model_num_layers = {"musicgen-small": 25, "musicgen-medium": 49, "musicgen-large": 49, "musicgen-encoder": 1, "jukebox": 72,
                    "baseline_mfcc": 1, "baseline_mel": 1, "baseline_chroma": 1, "baseline_concat": 1}
#model_num_layers = {"musicgen-small": 24, "musicgen-medium": 48, "musicgen-large": 48, "musicgen-encoder": 1, "jukebox": 1, "jukebox36": 1, "jukebox38": "jukebox38"} #until we get all jukebox layers

# because jukebox does a weird 1-indexing thing
jukebox_range = list(range(1,73))

act_layer_dim = {"musicgen-small_hidden": 1024, "musicgen-medium_hidden": 1536, "musicgen-large_hidden": 2048,
                   "musicgen-small_attn": 16, "musicgen-medium_attn": 24, "musicgen-large_attn": 32,
                   "musicgen-encoder": 128,
                 "jukebox": 4800,
                 "jukebox36": 4800,
                 "jukebox38": 4800,
                 "baseline_mfcc": 120,
                 "baseline_mel": 768,
                 "baseline_chroma": 72,
                 "baseline_concat": 960
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


# copying from https://github.com/brown-palm/syntheory/blob/main/dataset/synthetic/midi_instrument.py
pitched_exclude_categories = set(['Percussive', 'Sound effects', 'Synth Effects'])
pitched_exclude_inst = set(['Timpani', 'Celesta', 'Glockenspiel', 'Tubular Bells', 'Tango Accordion', 'Shakuhachi', 'Whistle', 'Orchestra Hit', 'Guitar harmonics', 'Harmonica', 'Accordion', 'Lead 7 (fifths)'])

pitched_inst_to_use = []

# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

def by_projpath(subpath=None,make_dir = False):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    if subpath != None:
        cur_path = os.path.join(cur_path, subpath)
        if os.path.exists(cur_path) == False and make_dir == True:
            os.makedirs(cur_path)
    return cur_path

emb_doc_dir = by_projpath(subpath='emb_doc', make_dir=True)

def get_baseline_shape_csv_path(cur_dataset):
    return os.path.join(emb_doc_dir, f'{cur_dataset}_baseline-shape.csv')

# https://stackoverflow.com/questions/2507808/how-to-check-whether-a-file-is-empty-or-not
def is_file_empty(file_handle):
    is_empty = False
    file_handle.seek(0, os.SEEK_END)
    if file_handle.tell():
        file_handle.seek(0)
    else:
        is_empty = True
    return is_empty

def get_shape_string(cur_shape, joiner='|'):
    return joiner.join([str(x) for x in cur_shape])
   
def parse_shape_string(shape_str, joiner = '|'):
    return tuple([int(x) for x in shape_str.split(joiner)])

def get_sorted_contents(cur_dir, is_relative = True):
    file_dir = None
    if is_relative == True:
        #file_dir = os.path.join(script_dir, cur_dir)
        file_dir = by_projpath(subpath=cur_dir, make_dir = False)
    else:
        file_dir = cur_dir
    files = [os.path.join(file_dir, x) for x in os.listdir(file_dir)]
    file_sort = sorted(files, key = os.path.getmtime)
    return file_sort

def remove_latest_file(cur_dir, is_relative = True):
    file_sort = get_sorted_contents(cur_dir, is_relative=is_relative)
    if len(file_sort) > 0:
        os.remove(file_sort[-1])
    return file_sort[:-1]

def get_basename(file, with_ext = True):
    if with_ext == True:
        return os.path.basename(file)
    else:
        return os.path.splitext(os.path.basename(file))[0]



def get_model_type(shorthand):
    model_lh = model_longhand.get(shorthand, shorthand)
    return model_type.get(model_lh, model_lh)

def get_layer_dim(shorthand):
    model_lh = model_longhand.get(shorthand, shorthand)
    return act_layer_dim[model_lh]

    
def get_embedding_num_layers(shorthand):
    longhand = model_longhand.get(shorthand, shorthand)
    mtype = model_type.get(longhand, longhand)
    num_layers = None
    if shorthand != 'mg_audio':
        num_layers = model_num_layers[mtype]
    else:
        num_layers = model_num_layers['musicgen-encoder']
    return num_layers

def old_get_embedding_shape(shorthand):
    longhand = model_longhand[shorthand]
    mtype = model_type[longhand]
    num_layers = model_num_layers[mtype]
    layer_dim = act_layer_dim[longhand]
    shape = (num_layers, layer_dim)
    return shape

def get_embedding_shape(shorthand):
    longhand = model_longhand.get(shorthand, shorthand)
    num_layers = get_embedding_num_layers(shorthand)
    layer_dim = act_layer_dim[longhand]
    shape = (num_layers, layer_dim)
    return shape
   

def get_model_act_path(model_shorthand, acts_folder = 'acts', dataset='polyrhythms', return_relative = False, make_dir = False):
    datapath = None
    if return_relative == False:
        actpath = by_projpath(acts_folder,make_dir = make_dir)
        datapath = os.path.join(actpath, dataset)
    else:
        datapath = acts_folder
    modelpath = os.path.join(datapath, model_shorthand)
    if os.path.exists(modelpath) == False and make_dir == True:
            os.makedirs(modelpath)
    return modelpath


def save_npy(save_arr, fname, model_shorthand, acts_folder = 'acts', dataset='polyrhythms', make_dir = True):
    modelpath = get_model_act_path(model_shorthand, acts_folder = acts_folder, dataset = dataset, return_relative = False, make_dir = make_dir)
    fpath = os.path.join(modelpath, fname)
    np.save(fpath, save_arr, allow_pickle = True)

# use_shape argument overrides shape getting (useful for baselines)
def get_embedding_file(model_shorthand, acts_folder = 'acts', dataset='polyrhythms', fname='', write = True, use_64bit = True, use_shape = None):
    modelpath = get_model_act_path(model_shorthand, acts_folder = acts_folder, dataset = dataset, return_relative = False, make_dir = write)
    fpath = os.path.join(modelpath, fname)
    fp = None
    dtype = 'float32'
    mode = 'r'
    shape = None
    if use_shape == None:
        shape = get_embedding_shape(model_shorthand)
    else:
        shape = use_shape
    if use_64bit == True:
        dtype = 'float64'
    if write == True:
        if os.path.isfile(fpath) == True:
            mode = 'r+'
        else:
            mode = 'w+'
    fp = np.memmap(fpath, dtype = dtype, mode=mode, order='C', shape=shape)
    return fp

def embedding_file_to_torch(model_shorthand, acts_folder = 'acts', dataset='polyrhythms', fname='', layer_idx = -1, use_64bit = True, device = 'cpu'):
    emb_file = get_embedding_file(model_shorthand, acts_folder = acts_folder, dataset=dataset, fname=fname, write = False, use_64bit = use_64bit)
    cur = None
    if layer_idx >= 0:
        cur = emb_file[layer_idx,:].copy()
    else:
        cur = emb_file.copy()
    v = torch.from_numpy(cur).to(device)
    return v

def npy_to_torch(model_shorthand, acts_folder = 'acts', dataset='polyrhythms', fname='', layer_idx = -1, use_64bit = False, device = 'cpu'):
    actpath = by_projpath(acts_folder)
    datapath = os.path.join(actpath, dataset)
    modelpath = os.path.join(datapath, model_shorthand)
    fpath = os.path.join(modelpath, fname)
    #arr = np.load(fpath, allow_pickle == True)
    arr = np.load(fpath)
    #print(arr.shape, layer_idx)
    cur = None
    if layer_idx >= 0 and len(arr.shape) > 1:
        cur = arr[layer_idx,:]
    else:
        cur = arr
    if cur.dtype == np.float32:
        if use_64bit == True:
            cur = cur.astype(np.float64)
    elif cur.dtype == np.float64:
        if use_64bit == False:
            cur = cur.astype(np.float32)
    return torch.from_numpy(cur).to(device)


with open(by_projpath('inst_list.csv'), 'r') as f:
    csvr = csv.reader(f, delimiter=',')
    for i,row in enumerate(csvr):
        if i > 0:
            inst_num = int(row[0].strip())
            inst_name = row[1].strip()
            inst_cat = row[2].strip()
            inst[inst_name] = {'program_number': inst_num, 'category': inst_cat}
            if inst_cat not in pitched_exclude_categories and inst_name not in pitched_exclude_inst:
                pitched_inst_to_use.append(inst_name)


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
    else:
        outname = f'{fsplit}'
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


def get_shape_from_df(cur_df, emb_type, fname=None):
    ret = tuple()
    match_shape = cur_df.filter(pl.col('emb_type').str.contains(emb_type) & pl.col('fname').str.contains(fname))['shape']
    if len(match_shape.shape) > 0:
        if match_shape.shape[0] > 0:         
            ret = parse_shape_string(match_shape[0], joiner = '|')
    return ret

