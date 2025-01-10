# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
import numpy as np
import librosa as lr
import util_hf as uhf
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
parser.add_argument("-at", "--activation_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
parser.add_argument("-lp", "--layers_per", type=int, default=4, help="layers per loop if doing all layers")
parser.add_argument("-l", "--layer_num", type=int, default=-1, help="layer num (all if < 0)")
parser.add_argument("-n", "--normalize", type=strtobool, default=True, help="normalize audio")

acts_folder = 'acts'
jb_model_sr = 44100
args = parser.parse_args()
lnum = args.layer_num
normalize = args.normalize
act_type = args.activation_type
dur = 4.0 
jb_dsamp_rate = 15
dataset = args.dataset
use_64bit = True

# exit if not a "real" dataset
if (dataset in um.all_datasets) == False:
    sys.exit('not a dataset')

path_list = um.path_list(os.path.join(dataset, 'wav'))

use_hf = dataset in um.hf_datasets
if use_hf == True:
    path_list = uhf.load_syntheory_train_dataset(_dataset)


def get_hf_audio(f, model_sr = 44100):
    audio, aud_sr = uhf.get_from_entry_syntheory_audio(f, mono=True, normalize =normalize, dur = dur)
    if aud_sr != model_sr:
        audio = librosa.resample(audio, orig_sr=aud_sr, target_sr=model_sr)
    return audio

def path_handler(f, using_hf=False, model_sr = 44100, logfile_handle=None):
    out_fname = None
    audio = None
    if using_hf == False:
        print(f'loading {f}', file=logfile_handle)
        out_fname = um.ext_replace(fpath, new_ext="pt")
    else:
        print(f"loading {f['audio']['path']}", file=lf)
        out_fname = um.ext_replace(f['audio']['path'], new_ext="pt")
    audio = None
    aud_sr = None
    if using_hf == True:
        audio, aud_sr = uhf.get_from_entry_syntheory_audio(f, mono=True, normalize =normalize, dur = dur)
        if aud_sr != model_sr:
            audio = librosa.resample(audio, orig_sr=aud_sr, target_sr=model_sr)
    return {'out_fname':out_fname, 'audio': audio}


# 1-indexed
def get_jukebox_layer_activations(fpath=None, audio = None, layers=list(range(1,73))):
    reps = None
    if fpath != None:
        acts = jml.extract(fpath=fpath, layers=layers, duration=dur, meanpool=True, downsample_target_rate=jb_dsamp_rate, downsample_method=None)
    else:
        acts = jml.extract(audio=audio, layers=layers, duration=dur, meanpool=True, downsample_target_rate=jb_dsamp_rate, downsample_method=None)
    jml.lib.empty_cache()
    return np.array([acts[i] for i in layers])

def get_activations(cur_pathlist, cur_act_type, cur_dataset, using_hf = False, logfile_handle=None):
    cur_model_type = um.get_model_type(cur_act_type)
    model_sr = um.model_sr[cur_model_type]
    if cur_model_type == 'jukebox':
        jml.setup_models(cache_dir='/nfs/guille/eecs_research/soundbendor/kwand/jukemirlib')
    for fidx,f in enumerate(cur_pathlist):
        fdict = path_handler(f, model_sr = model_sr, using_hf = using_hf, logfile_handle=logfile_handle)
        #outpath = os.path.join(out_dir, outname)
        out_fname = fdict['out_fname']
        act_file = um.get_activations_file(cur_act_type, acts_folder=acts_folder, dataset=cur_dataset, fname=out_fname, use_64bit = use_64bit, write=True)


