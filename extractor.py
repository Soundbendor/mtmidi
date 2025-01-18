# heavily referencing the original code:
# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
import sys,os,time,argparse
import numpy as np
import torch
import librosa as lr
import util_hf as uhf
import util as um
import jukemirlib as jml
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from distutils.util import strtobool

acts_folder = 'acts'
jb_dsamp_rate = 15
dur = 4.0 



def get_hf_audio(f, model_sr = 44100, normalize=True):
    audio, aud_sr = uhf.get_from_entry_syntheory_audio(f, mono=True, normalize =normalize, dur = dur)
    if aud_sr != model_sr:
        audio = lr.resample(audio, orig_sr=aud_sr, target_sr=model_sr)
    return audio

def path_handler(f, using_hf=False, model_sr = 44100, wav_path = None, model_type = 'jukebox', dur = 4., normalize = True, out_ext = 'dat', logfile_handle=None):
    out_fname = None
    audio = None
    in_dir = um.by_projpath(wav_path)
    in_fpath = None
    out_fname = None
    fname = None
    if using_hf == False:
        print(f'loading {f}', file=logfile_handle)
        in_fpath = os.path.join(in_dir, f)
        out_fname = um.ext_replace(f, new_ext=out_ext)
        fname = um.ext_replace(f, new_ext='')
        # don't need to load audio if jukebox
        if model_type != 'jukebox':
            audio = um.load_wav(f, dur = dur, normalize = normalize, sr = model_sr,  load_dir = in_dir)
    else:
        hf_path = f['audio']['path']
        print(f"loading {hf_path}", file=lf)
        out_fname = um.ext_replace(hf_path, new_ext=out_ext)
        fname = um.ext_replace(hf_path, new_ext='')
    aud_sr = None
    if using_hf == True:
        audio, aud_sr = uhf.get_from_entry_syntheory_audio(f, mono=True, normalize =normalize, dur = dur)
        if aud_sr != model_sr:
            audio = lr.resample(audio, orig_sr=aud_sr, target_sr=model_sr)
    return {'in_fpath': in_fpath, 'out_fname': out_fname, 'audio': audio, 'fname': fname}

def get_musicgen_encoder_embeddings(model, proc, audio, meanpool = True, model_sr = 32000, device='cpu'):
    procd = proc(audio = audio, sampling_rate = model_sr, padding=True, return_tensors = 'pt')
    procd.to(device)
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
    return out.detach().cpu().numpy()

def get_musicgen_lm_hidden_states(model, proc, audio, text="", meanpool = True, model_sr = 32000, device = 'cpu'):
    procd = proc(audio = audio, text = text, sampling_rate = model_sr, padding=True, return_tensors = 'pt')
    procd.to(device)
    outputs = model(**procd, output_attentions=True, output_hidden_states=True)
    dhs = None
    
    #dat = None

    # hidden
    # outputs is a tuple of tensors with  shape (batch_size, seqlen, dimension) with 1 per layer
    # torch stack makes it so we have (num_layers, batch_size, seqlen, dimension)
    # then we average over seqlen in the meanpool case
    # then squeeze to get rid of the 1 dim (if batch_size == 1)
    # final shape is (num_layers, batch_size, dim)  (or (num_layers, dim) if bs=1)
    
    # attentions
    # outputs is a tuple of tensors with  shape (batch_size, num_heads, seqlen, seqlen) with 1 per layer
    # torch stack makes it so we have (num_layers, batch_size, num_heads, seqlen, sequlen)
    # then we average over seqlens in the meanpool case
    # then squeeze to get rid of the 1 dim (if batch_size == 1)
    # final shape is (num_layers, batch_size, num_heads) (or (num_layers, num_heads) if bs = 1)

    if meanpool == True:
        dhs = torch.stack(outputs.decoder_hidden_states).mean(axis=2).squeeze()
        #dat = torch.stack(outputs.decoder_attentions).mean(axis=(3,4)).squeeze()
    else:
        dhs = torch.stack(outputs.decoder_hidden_states).squeeze()
        #dat = torch.stack(outputs.decoder_attentions).squeeze()
    return dhs.detach().cpu().numpy()




# 1-indexed
def get_jukebox_layer_embeddings(fpath=None, audio = None, layers=list(range(1,73))):
    reps = None
    if fpath != None:
        acts = jml.extract(fpath=fpath, layers=layers, duration=dur, meanpool=True, downsample_target_rate=jb_dsamp_rate, downsample_method=None)
    else:
        acts = jml.extract(audio=audio, layers=layers, duration=dur, meanpool=True, downsample_target_rate=jb_dsamp_rate, downsample_method=None)
    jml.lib.empty_cache()
    return np.array([acts[i] for i in layers])

def get_embeddings(cur_act_type, cur_dataset, layers_per = 4, layer_num = -1, normalize = True, dur = 4., use_64bit = True, logfile_handle=None, recfile_handle = None):
    cur_model_type = um.get_model_type(cur_act_type)
    model_sr = um.model_sr[cur_model_type]
    model_longhand = um.model_longhand[cur_act_type]
    
    using_hf = cur_dataset in um.hf_datasets
    # musicgen stuff
    device = 'cpu'
    num_layers = None
    proc = None
    model = None
    text = ""
    wav_path = os.path.join(um.by_projpath('wav'), cur_dataset)
    cur_pathlist = None
    if using_hf == True:
        cur_pathlist = uhf.load_syntheory_train_dataset(cur_dataset)
    else:
        cur_pathlist = um.path_list(wav_path)

    if cur_model_type == 'jukebox':
        jml.setup_models(cache_dir='/nfs/guille/eecs_research/soundbendor/kwand/jukemirlib')
    # is a musicgen model, need to specify torch params (?)
    else:

        if torch.cuda.is_available() == True:
            device = 'cuda'
            torch.cuda.empty_cache()
            torch.set_default_device(device)


        model_str = f"facebook/{cur_model_type}" 

        proc = AutoProcessor.from_pretrained(model_str)
        model = MusicgenForConditionalGeneration.from_pretrained(model_str, device_map=device)
        model_sr = model.config.audio_encoder.sampling_rate

    print('file,is_extracted', file=rf)
    for fidx,f in enumerate(cur_pathlist):
        fdict = path_handler(f, model_sr = model_sr, wav_path = wav_path, normalize = normalize, dur = dur,model_type = cur_model_type, using_hf = using_hf, logfile_handle=logfile_handle)
        #outpath = os.path.join(out_dir, outname)
        out_fname = fdict['out_fname']
        fpath = fdict['in_fpath']
        audio = fdict['audio']
        # store by cur_act_type (model shorthand)
        emb_file = um.get_embedding_file(cur_act_type, acts_folder=acts_folder, dataset=cur_dataset, model_folder = cur_act_type, fname=out_fname, use_64bit = use_64bit, write=True)
        if cur_model_type == 'jukebox':
            print(f'--- extracting jukebox for {f} with {layers_per} layers at a time ---', file=logfile_handle)
            # note that layers are 1-indexed in jukebox
            # so let's 0-idx and then add 1 when feeding into jukebox fn
            layer_gen = (list(range(l, min(um.model_num_layers['jukebox'], l + layers_per))) for l in range(0,um.model_num_layers['jukebox'], layers_per)) 
            if layer_num > 0:
                # 0-idx from 1-idxed argt
                layer_gen = ([l-1] for l in [layer_num])
            for layer_arr in layer_gen:
                # 1-idx for passing into fn
                j_idx = [l+1 for l in layer_arr]
                print(f'extracting layers {j_idx}', file=logfile_handle)
                rep_arr = get_jukebox_layer_embeddings(fpath=fpath, audio = audio, layers=j_idx)
                emb_file[layer_arr,:] = rep_arr
                emb_file.flush()
        else:
            audio_ipt = fdict['audio']
            if model_longhand == "musicgen-encoder":
                print(f'--- extracting musicgen-encoder for {f} ---', file=logfile_handle)

                rep_arr = get_musicgen_encoder_embeddings(model, proc, audio_ipt, meanpool = True, model_sr = model_sr, device=device)
                emb_file[:,:] = rep_arr
                emb_file.flush()
            else:

                print(f'--- extracting musicgen_lm for {f} ---', file=logfile_handle)
                rep_arr =  get_musicgen_lm_hidden_states(model, proc, audio_ipt, text="", meanpool = True, model_sr = model_sr, device=device)
                emb_file[:,:] = rep_arr
                emb_file.flush()
        fname = fdict['fname']
        print(f'{fname},1', file=rf)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ub", "--use_64bit", type=strtobool, default=True, help="use 64-bit")
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-at", "--activation_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-lp", "--layers_per", type=int, default=4, help="layers per loop if doing all layers (for jukebox)")
    parser.add_argument("-l", "--layer_num", type=int, default=-1, help="1-indexed layer num (all if < 0, for jukebox)")
    parser.add_argument("-n", "--normalize", type=strtobool, default=True, help="normalize audio")


    args = parser.parse_args()
    use_64bit = args.use_64bit
    lnum = args.layer_num
    lper = args.layers_per
    normalize = args.normalize
    act_type = args.activation_type
    dataset = args.dataset
    # exit if not a "real" dataset
    logdir = um.by_projpath(subpath='log', make_dir = True)
    timestamp = int(time.time() * 1000)

    # miscellaneous logs
    log_fname = f'{dataset}_{act_type}-{timestamp}.log'
    # record saving activations
    rec_fname = f'{dataset}_{act_type}-{timestamp}.csv'
    if normalize == True:
        log_fname = f'{dataset}_{act_type}_norm-{timestamp}.txt'
        rec_fname = f'{dataset}_{act_type}_norm-{timestamp}.csv'
    log_fpath = os.path.join(logdir, log_fname)
    rec_fpath = os.path.join(logdir, rec_fname)
    if (dataset in um.all_datasets) == False:
        sys.exit('not a dataset')
    else:
        lf = open(log_fpath, 'a')
        rf = open(rec_fpath, 'w')
        print(f'=== running extraction for {dataset} with {act_type} at {timestamp} ===', file=lf)
        get_embeddings(act_type, dataset, layers_per = lper, layer_num = lnum, normalize = normalize, dur = dur, use_64bit = use_64bit, logfile_handle=lf, recfile_handle=rf)
        lf.close()
        rf.close()
