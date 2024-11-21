# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
# https://huggingface.co/docs/transformers/main/model_doc/musicgen
# https://huggingface.co/docs/transformers/main/en/model_doc/encodec#transformers.EncodecFeatureExtractor
import os
import torch
import librosa
import util as um
import argparse
from transformers import AutoProcessor, MusicenForConditionalGeneration

sr = 32000
model_num_layers = {"facebook/musicgen-small": 24, "facebook/musicgen-medium": 48,
              "facebook/musicgen-large": 48}
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--size", type=str, default="small", help="audio,small, medium, large")
args.parser.parse_args()
model_size = args.size
model_str = "facebook/musicgen-small"
emb_dir = "mg_small"

if model_size == "medium":
    model_str = "facebook/musicgen-medium"
    emb_dir = "mg_medium"
elif model_size == "large":

    model_str = "facebook/musicgen-large"
    emb_dir = "mg_large"
elif model_size == "audio":

    model_str = "facebook/musicgen-large"
    emb_dir = "mg_audio"
num_layers = model_num_layers[model_str]
if model_size == "audio":
    num_layers = -1
out_dir = um.by_projpath(os.path.join('acts', emb_dir), make_dir = True)
load_dir = um.by_projpath('wav')
log = um.by_projpath(os.path.join('log', 'jml.log'))
#dsamp_rate = 22050
dsamp_rate = 15
layer_act = 36
dur = 4.0

proc = AutoProcessor.from_pretrained(model_str)
model = MusicgenForConditionalGeneration.from_pretrained(model_str)
#layer_acts = [x for x in range(1,73)]
if os.path.isfile(log):
    os.remove(log)
with open(log, 'a') as lf:
    for fidx,f in enumerate(os.listdir(load_dir)):
        #if fidx > 0: break
        procd = proc(raw_audio=audio, sampling_rate=sr, padding=True, return_tensors="pt")
        fsplit = '.'.join(f.split('.')[:-1])
        outname = f'{fsplit}.pt'
        outpath = os.path.join(out_dir, outname)

        fpath = os.path.join(load_dir, f)
        print(f'loading {fpath}', file=lf)
        reps = jml.extract(fpath = fpath, layers=[layer_act], duration= dur, downsample_method=None, downsample_target_rate=dsamp_rate, meanpool = True)
        torch.save(reps[layer_act], outpath)
        jml.lib.empty_cache()
                   
