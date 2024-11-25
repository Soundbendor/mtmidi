# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
# https://huggingface.co/docs/transformers/main/model_doc/musicgen
# https://huggingface.co/docs/transformers/main/en/model_doc/encodec#transformers.EncodecFeatureExtractor
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/musicgen/modeling_musicgen.py
import os
import torch
import librosa
import util as um
import argparse
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from distutils.util import strtobool

# outputs of lm are all tuples where each entry is a tensor
# sizes:
# decoder.hidden_states
# --- large: 1, 200, 2048 (49 hidden states, 0-47: AddBackward0, 48: NativeLayerNormBackward0)
# --- medium: 1, 200, 1536 (49 hidden states, 0-47: AddBackward0, 48: NativeLayerNormBackward0)
# --- small: 1, 200, 1024 (25 hidden states, 0-47: AddBackward0, 48: NativeLayerNormBackward0)
# decoder.attention
# --- large: 1, 32, 200, 200 (48, all ViewBackward0)
# --- medium: 1, 24, 200, 200 (48, all ViewBackward0)
# --- medium: 1, 16, 200, 200 (24, all ViewBackward0)

sr = 32000
model_num_layers = {"facebook/musicgen-small": 24, "facebook/musicgen-medium": 48,
              "facebook/musicgen-large": 48}
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--size", type=str, default="small", help="audio,small, medium, large")
parser.add_argument("-n", "--normalize", type=strtobool, default=True, help="normalize audio")
parser.add_argument("-d", "--debug", type=strtobool, default=False, help="debug")
parser.add_argument("-m", "--meanpool", type=strtobool, default=True, help="to meanpool")
parser.add_argument("-hl", "--save_hidden", type=strtobool, default=True, help="save hidden states")
parser.add_argument("-at", "--save_attn", type=strtobool, default=False, help="save attention")

args = parser.parse_args()
model_size = args.size
normalize = args.normalize
debug = args.debug
meanpool = args.meanpool
save_hidden = args.save_hidden
save_attn = args.save_attn

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
out_dir = None
if meanpool == True:
    out_dir = um.by_projpath(os.path.join('acts', f'{emb_dir}_mp'), make_dir = True)
else:
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
#proc = AutoProcessor.from_pretrained(model_str, device_map = device)
#model = MusicgenForConditionalGeneration.from_pretrained(model_str, device_map = device)
proc = AutoProcessor.from_pretrained(model_str)
#model = MusicgenForConditionalGeneration.from_pretrained(model_str)
model = MusicgenForConditionalGeneration.from_pretrained(model_str, device_map=device)
#proc.to(device)
#model.to(device)
sr = model.config.audio_encoder.sampling_rate
#layer_acts = [x for x in range(1,73)]
if os.path.isfile(log):
    os.remove(log)
with open(log, 'a') as lf:
    for fidx,f in enumerate(os.listdir(load_dir)):
        if debug == True:
            if fidx > 0: break
        
        dhs = None
        dhs_mp = None
        dat = None
        dat_mp = None
        audio = None

        procd = None
        audio = um.load_wav(f, dur = dur, normalize = normalize, sr = sr,  load_dir = load_dir)
        if model_size == 'audio':
            procd = proc(audio = audio, sampling_rate = sr, padding=True, return_tensors = 'pt')
        else:
            procd = proc(audio = audio, text = text, sampling_rate = sr, padding=True, return_tensors = 'pt')

        procd.to(device)
        fsplit = '.'.join(f.split('.')[:-1])
        outname = f'{fsplit}.pt'
        outpath = os.path.join(out_dir, outname)

        fpath = os.path.join(load_dir, f)
        print(f'loading {fpath}', file=lf)

        if model_size == 'audio':
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
                out_mp = torch.mean(out,axis=2).squeeze()
                torch.save(out_mp, outpath)
            else:
                # still need to squeeze
                # gives shape (128, 200)
                out_save = out.squeeze()
                torch.save(out_save, outpath)

            if debug == True:
                encprops = dir(enc)
                iptprops = dir(procd)
                print("processed properties", file=lf)
                print(iptprops, file=lf)
                
                print("encoder properties", file=lf)
                print(encprops, file=lf)

                encprops2 = dir(enc.encoder)
                print("encoder properties 2", file=lf)
                print(encprops2, file=lf)

                numlayers = len(enc.encoder.layers)
                print(f"num layers: {numlayers}", file=lf)

                
                # going from the audio encder
                out2 = enc.encode(**procd)

                print("iterating through layers", file=lf)
                print(out.shape, file=lf)
               
                print("out2 propeties", file=lf)
                out2prop = dir(out2)
                print(out2prop, file=lf)

                print("out2 outputs: audio_scales",file=lf)
                print(out2.audio_scales,file=lf)

                print("out2 outputs: audio_codes",file=lf)
                print(out2.audio_codes, file=lf)

                out3 = enc(**procd)


                print("out3 propeties", file=lf)
                out3prop = dir(out3)
                print(out3prop, file=lf)

                out3av = out3['audio_values']
                avshape = out3av.shape
                print(f"out3 output: audio_values ({avshape})", file=lf)
                print(out3av, file=lf)
                procd2 = proc(audio = audio, text = text, sampling_rate = sr, padding=True, return_tensors = 'pt')
                procd2.to(device)
                outputs = model(**procd2, output_attentions=True, output_hidden_states=True)
                enc_lh = outputs.encoder_last_hidden_state
                print("last hidden state of encoder", file=lf)
                print(enc_lh.shape, file=lf)

                print("iteration output", file=lf)
                print(out, file=lf)

                print("last hidden state output", file=lf)
                print(enc_lh, file=lf)


                enc_h = outputs.encoder_hidden_states
                enc_at = outputs.encoder_attentions
                enc_h_sz = len(enc_h)
                enc_at_sz = len(enc_at)
                print(f'encoder hidden states: {enc_h_sz}', file=lf)
                for i in range(enc_h_sz):
                    print(f'----{i}----', file=lf)
                    print(enc_h[i].shape, file=lf)
                    print(enc_h[i].grad_fn, file=lf)
                
                print(f'encoder hidden states output: {enc_h_sz}', file=lf)
                for i in range(enc_h_sz):
                    print(f'----{i}----', file=lf)
                    print(enc_h[i], file=lf)
                

                print(f'encoder attentions: {enc_at_sz}', file=lf)
                for i in range(enc_at_sz):
                    print(f'----{i}----', file=lf)
                    print(enc_at[i].shape, file=lf)
                    print(enc_at[i].grad_fn, file=lf)
                
                print(f'encoder attentions output: {enc_at_sz}', file=lf)
                for i in range(enc_at_sz):
                    print(f'----{i}----', file=lf)
                    print(enc_at[i], file=lf)
                



        else:
            outputs = model(**procd, output_attentions=True, output_hidden_states=True)
            dhs = torch.vstack(outputs.decoder_hidden_states)
            dat = torch.vstack(outputs.decoder_attentions)
           
            if meanpool == True:
                # gives shape (24/48, 1024/1536/2048)
                if save_hidden == True:
                    dhs_mp = torch.mean(dhs,axis=1)
                    torch.save(dhs_mp, outpath)
                # gives shape (16/24/32)
                if save_attn == True:
                    dat_mp = torch.mean(dat,axis=(2,3))
                    torch.save(dat_mp, outpath)
            else:
                # gives shape (24/48, 200, 1024/1536/2048)
                if save_hidden == True:
                    torch.save(dhs, outpath)
                # gives shape (16/24/32, 200, 200)
                if save_attn == True:
                    torch.save(dat, outpath)
            if debug == True:
                dhs_sz = len(dhs)
                print(f'hidden states: {dhs_sz}', file=lf)
                for i in range(dhs_sz):
                    print(f'----{i}----', file=lf)
                    print(dhs[i].shape, file=lf)
                    print(dhs[i].grad_fn, file=lf)
                
                dat_sz = len(dat)
                print(f'attention: {dat_sz}', file=lf)
                for i in range(dat_sz):
                    print(f'----{i}----', file=lf)
                    print(dat[i].shape, file=lf)
                    print(dat[i].grad_fn, file=lf)

            #torch.save(reps[layer_act], outpath)
            #jml.lib.empty_cache()
                       
