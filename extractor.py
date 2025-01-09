# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
import numpy as np
import librosa as lr
import util_hf as uhf

parser.add_argument("-at", "--activation_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
parser.add_argument("-lp", "--layers_per", type=int, default=4, help="layers per loop if doing all layers")
parser.add_argument("-l", "--layer_num", type=int, default=-1, help="layer num (all if < 0)")
parser.add_argument("-n", "--normalize", type=strtobool, default=True, help="normalize audio")
parser.add_argument("-hf", "--hf_dataset", type=str, default="", help="use old hugging face dataset if passed")

jb_model_sr = 44100
args = parser.parse_args()
lnum = args.layer_num
hfds_str = args.hf_dataset
normalize = args.normalize
acts_dir = 'acts'
path_list = um.path_list('wav')
dur = 4.0 
use_hf = len(hfds_str) > 0
jb_dsamp_rate = 15


def get_hf_audio(f, model_sr = 44100):
    audio, aud_sr = uhf.get_from_entry_syntheory_audio(f, mono=True, normalize =normalize, dur = dur)
    if aud_sr != model_sr:
        audio = librosa.resample(audio, orig_sr=aud_sr, target_sr=model_sr)
    return audio

# 1-indexed
def get_jukebox_activations(fpath=None, audio = None, layers=list(range(1,73))):
    reps = None
    if fpath != None:
        acts = jml.extract(fpath=fpath, layers=layers, duration=dur, meanpool=True, downsample_target_rate=jb_dsamp_rate, downsample_method=None)
    else:
        acts = jml.extract(audio=audio, layers=layers, duration=dur, meanpool=True, downsample_target_rate=jb_dsamp_rate, downsample_method=None)
    return np.array([acts[i] for i in layers])
