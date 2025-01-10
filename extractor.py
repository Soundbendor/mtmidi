# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
import numpy as np
import torch
import librosa as lr
import util_hf as uhf
import argparse
from transformers import AutoProcessor, MusicgenForConditionalGeneration

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
parser.add_argument("-at", "--activation_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
parser.add_argument("-lp", "--layers_per", type=int, default=4, help="layers per loop if doing all layers (for jukebox)")
parser.add_argument("-l", "--layer_num", type=int, default=-1, help="1-indexed layer num (all if < 0, for jukebox)")
parser.add_argument("-n", "--normalize", type=strtobool, default=True, help="normalize audio")

acts_folder = 'acts'
jb_model_sr = 44100
args = parser.parse_args()
lnum = args.layer_num
lper = args.layer_per
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

def path_handler(f, using_hf=False, model_sr = 44100, wav_path = None, model_type = 'jukebox', dur = 4., normalize = True, out_ext = 'dat', logfile_handle=None):
    out_fname = None
    audio = None
    in_dir = um.by_projpath(wav_path)
    in_fpath = None
    out_fname = None
    if using_hf == False:
        print(f'loading {f}', file=logfile_handle)
        in_fpath = os.path.join(in_dir, f)
        out_fname = um.ext_replace(f, new_ext=out_ext)
        # don't need to load audio if jukebox
        if model_type != 'jukebox':
            audio = um.load_wav(f, dur = dur, normalize = normalize, sr = model_sr,  load_dir = in_dir)
    else:
        print(f"loading {f['audio']['path']}", file=lf)
        out_fname = um.ext_replace(f['audio']['path'], new_ext=out_ext)
    audio = None
    aud_sr = None
    if using_hf == True:
        audio, aud_sr = uhf.get_from_entry_syntheory_audio(f, mono=True, normalize =normalize, dur = dur)
        if aud_sr != model_sr:
            audio = librosa.resample(audio, orig_sr=aud_sr, target_sr=model_sr)
    return {'in_fpath': in_fpath, 'out_fname': out_fname, 'audio': audio}

def get_musicgen_encoder_activations(model, proc, audio, meanpool = True, model_sr = 32000):
    procd = proc(audio = audio, sampling_rate = model_sr, padding=True, return_tensors = 'pt')
    enc = model.get_audio_encoder()
    out = procd['input_values']
    
    # iterating through layers as in original syntheory codebase
    # https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
    for layer in enc.encoder.layers:
        out = layer(out)

    # output shape, (1, 128, 200), where 200 are the timesteps
    # so average across timesteps for max pooling


    if meanpool == True:
        # gives shape (128)
        out = torch.mean(out,axis=2).squeeze()
    else:
        # still need to squeeze
        # gives shape (128, 200)
        out = out.squeeze()
    return out


# 1-indexed
def get_jukebox_layer_activations(fpath=None, audio = None, layers=list(range(1,73))):
    reps = None
    if fpath != None:
        acts = jml.extract(fpath=fpath, layers=layers, duration=dur, meanpool=True, downsample_target_rate=jb_dsamp_rate, downsample_method=None)
    else:
        acts = jml.extract(audio=audio, layers=layers, duration=dur, meanpool=True, downsample_target_rate=jb_dsamp_rate, downsample_method=None)
    jml.lib.empty_cache()
    return np.array([acts[i] for i in layers])

def get_activations(cur_pathlist, cur_act_type, cur_dataset, using_hf = False, layers_per = 4, layer_num = -1, normalize = True, dur = 4., wav_path=None, logfile_handle=None):
    cur_model_type = um.get_model_type(cur_act_type)
    model_sr = um.model_sr[cur_model_type]
    model_longhand = um.model_longhand(cur_act_type)
    
    device = 'cpu'
    if cur_model_type == 'jukebox':
        jml.setup_models(cache_dir='/nfs/guille/eecs_research/soundbendor/kwand/jukemirlib')
    # is a musicgen model, need to specify torch params (?)
    else:

        if torch.cuda.is_available() == True:
            device = 'cuda'
            torch.cuda.empty_cache()
            torch.set_default_device(device)


    for fidx,f in enumerate(cur_pathlist):
        fdict = path_handler(f, model_sr = model_sr, wav_path = wav_path, normalize = normalize, dur = dur,model_type = cur_model_type, using_hf = using_hf, logfile_handle=logfile_handle)
        #outpath = os.path.join(out_dir, outname)
        out_fname = fdict['out_fname']
        fpath = fdict['in_fpath']
        audio = fdict['audio']
        # store by cur_act_type (model shorthand)
        act_file = um.get_activations_file(cur_act_type, acts_folder=acts_folder, dataset=cur_dataset, model_folder = cur_act_type, fname=out_fname, use_64bit = use_64bit, write=True)
        if cur_model_type == 'jukebox':
            # note that layers are 1-indexed in jukebox
            # so let's 0-idx and then add 1 when feeding into jukebox fn
            layer_gen = (list(range(l, l + layers_per)) for l in range(0,um.model_num_layers['jukebox'], layers_per)) 
            if layer_num > 0:
                # 0-idx from 1-idxed argt
                layer_gen = ([l-1] for l in [layer_num])
            for layer_arr in layer_gen:
                # 1-idx for passing into fn
                j_idx = [l+1 for l in layer_arr]
                rep_arr = get_jukebox_layer_activations(fpath=fpath, audio = audio, layers=j_idx)
                act_file[layer_arr,:] = rep_arr
        else:
            text = ""
            model_str = f"facebook/{cur_model_type}" 
            num_layers = model_num_layers[model_str]

            proc = AutoProcessor.from_pretrained(model_str)
            model = MusicgenForConditionalGeneration.from_pretrained(model_str, device_map=device)
            model_sr = model.config.audio_encoder.sampling_rate

            if model_longhand == "musicgen-encoder":

                rep_arr = get_musicgen_encoder_activations(model, proc, fdict['audio'], meanpool = True, model_sr = model_sr)
                act_file = rep_arr
            else:
                pass
