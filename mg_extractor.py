# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
# https://huggingface.co/docs/transformers/main/model_doc/musicgen
# https://huggingface.co/docs/transformers/main/en/model_doc/encodec#transformers.EncodecFeatureExtractor
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/musicgen/modeling_musicgen.py
import os
import torch
import librosa
import util as um
import argparse
from transformers import AutoProcessor, MusicenForConditionalGeneration
from distutils.util import strtobool



sr = 32000
model_num_layers = {"facebook/musicgen-small": 24, "facebook/musicgen-medium": 48,
              "facebook/musicgen-large": 48}
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--size", type=str, default="small", help="audio,small, medium, large")
parser.add_argument("-n", "--normalize", type=strtobool, default=True, help="normalize audio")
args.parser.parse_args()
model_size = args.size
normalize = args.normalize

text = ""
model_str = "facebook/musicgen-small"
emb_dir = "mg_small"
log_path = "mg_medium"
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

device = 'cpu'
log_path = emb_dir

if torch.cuda.is_available() == True:
    device = 'cuda'
    torch.cuda.empty_cache()
    torch.set_default_device(device)


if model_size == "audio":
    num_layers = -1
out_dir = um.by_projpath(os.path.join('acts', emb_dir), make_dir = True)
load_dir = um.by_projpath('wav')
log = um.by_projpath(os.path.join('log', f'{log_path}.log'))
#dsamp_rate = 22050
layer_act = 36
dur = 4.0

# https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/musicgen

# ----- language model -----

# decoder_hidden_states --- 
# output_hidden_states = True or config.output_hidden_states = True
# tuple of torch.floattensor: (output of embeddings, one output of each layer of following shape)
# shape: (0: batch_size, 1: sequence_length, 2:hidden_size)

# "hidden states of the decoder at the output of each layer + initial embedding inputs"

# decoder_attentions ---
# output_attentions = True or config.output_attentions = True
# tuple of torch.floattensor (one per layer) with followng shape
# shape: batch_size, num_heads, sequence_length, sequence_length

# "attentions weights of decoder, after attention softmax, used to compute the weighted average in the self-attention heads"

# ---- encodec ------
 # https://github.com/huggingface/transformers/blob/main/src/transformers/models/musicgen/modeling_musicgen.py
 
# get_audio_encoder returns self.audio_encoder 
proc = AutoProcessor.from_pretrained(model_str)
model = MusicgenForConditionalGeneration.from_pretrained(model_str)
sr = model.config.audio_encoder.sampling_rate
#layer_acts = [x for x in range(1,73)]
if os.path.isfile(log):
    os.remove(log)
with open(log, 'a') as lf:
    for fidx,f in enumerate(os.listdir(load_dir)):
        if fidx > 0: break
        procd = None
        audio = um.load_wav(f, dur = dur, normalize = normalize, sr = sr,  load_dir = load_dir)
        if model_size == 'audio':
            procd = proc(audio = audio, sampling_rate = sr, padding=True, return_tensors = 'pt')
        else:
            procd = proc(audio = audio, text = text, sampling_rate = sr, padding=True, return_tensors = 'pt')

        fsplit = '.'.join(f.split('.')[:-1])
        outname = f'{fsplit}.pt'
        outpath = os.path.join(out_dir, outname)

        fpath = os.path.join(load_dir, f)
        print(f'loading {fpath}', file=lf)
        outputs = model(**procd, output_attentions=True, output_hidden_states=True)
        dhs = outputs.decoder_hidden_states
        dat = outputs.decoder_attentions
        hs_shape = dhs.shape
        dat_shape = dat.shape
        print(f'hidden state shape: {hs_shape}, attention shape: {dat_shape}')
        #torch.save(reps[layer_act], outpath)
        #jml.lib.empty_cache()
                   
