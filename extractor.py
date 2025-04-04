# heavily referencing the original code:
# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
import sys,os,time,argparse, copy
import numpy as np
import torch
import librosa as lr
from librosa import feature as lrf
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

# mel = melspectrogram, chroma = chroma_cqt, mfcc = mfcc
# concat = mel + chroma + mfcc
def get_baseline_features(audio, sr=22050, feat_type="concat"):
    feat = []
    if feat_type == "baseline_mel" or feat_type == "baseline_concat":
        # mel spectrogram
        # https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
        cur_mel = lrf.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length = 512)
        # returns (N=1, n_mels, t)
        feat.append(cur_mel)
    if feat_type == "baseline_chroma" or feat_type == "baseline_concat":
        # constant q chromagram
        # https://librosa.org/doc/0.10.2/generated/librosa.feature.chroma_cqt.html#librosa.feature.chroma_cqt
        # default fmin = 32.7
        # default norm = infinity norm normalization
        # default 36 bins per octave
        cur_chroma = lrf.chroma_cqt(y=audio, sr=sr, hop_length=512)
        # returns (N=1, n_chroma, t)
        feat.append(cur_chroma)
    if feat_type == "baseline_mfcc" or feat_type == "baseline_concat":
        # mfcc
        # https://librosa.org/doc/0.10.2/generated/librosa.feature.mfcc.html#librosa.feature.mfcc
        # default 20 mfccs
        # default orthonormal dct basis
        cur_mfcc = lrf.mfcc(y=audio, sr = sr, n_mfcc = 20)
        # returns (N=1, n_mfcc, t)
        feat.append(cur_mfcc)
    ft_vec = None
    for ft_idx,ft in enumerate(feat):
        # as in the original codebase, do 0,1,2-order diff across time dimension
        # and then take mean and std dev across time dimension
        # note that 0 order diff is just the same array
        for diff_n in range(3):
            cur_diff = np.diff(ft, n=diff_n, axis=1)
            cur_mean = np.mean(cur_diff, axis=1)
            cur_std = np.std(cur_diff, axis=1)
            cur = np.concatenate((cur_mean, cur_std))
            if diff_n == 0 and ft_idx == 0:
                ft_vec = copy.deepcopy(cur)
            else:
                ft_vec = np.concatenate([ft_vec, copy.deepcopy(cur)])
    # make it a 1 x cur_dim vector for consistency (i think)
    if len(ft_vec.shape) < 2:
        ft_vec = np.expand_dims(ft_vec,axis=0)
    return ft_vec


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

def get_embeddings(cur_act_type, cur_dataset, layers_per = 4, layer_num = -1, normalize = True, dur = 4., use_64bit = True, logfile_handle=None, recfile_handle = None, memmap = True, pickup = False):
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
    out_ext = 'dat'
    if memmap == False:
        out_ext = 'npy'
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

    #print('file,is_extracted', file=rf)

    # existing files removing latest (since it may be partially written) and removing extension for each of checking
    existing_name_set = None
    if pickup == True:
        _file_dir = um.get_model_act_path(cur_act_type, acts_folder = acts_folder, dataset=cur_dataset, return_relative = False, make_dir = False)
        existing_files = um.remove_latest_file(_file_dir, is_relative = False)
        existing_name_set = set([um.get_basename(_f, with_ext = False) for _f in existing_files])
    for fidx,f in enumerate(cur_pathlist):
        if pickup == True:
            cur_name = um.get_basename(f, with_ext = False)
            if cur_name in existing_name_set:
                continue
        fdict = path_handler(f, model_sr = model_sr, wav_path = wav_path, normalize = normalize, dur = dur,model_type = cur_model_type, using_hf = using_hf, logfile_handle=logfile_handle, out_ext = out_ext)
        #outpath = os.path.join(out_dir, outname)
        out_fname = fdict['out_fname']
        fpath = fdict['in_fpath']
        audio = fdict['audio']
        # store by cur_act_type (model shorthand)
        emb_file = None
        np_arr = None
        if memmap == True:
            emb_file = um.get_embedding_file(cur_act_type, acts_folder=acts_folder, dataset=cur_dataset, fname=out_fname, use_64bit = use_64bit, write=True, use_shape = None)
        if cur_model_type == 'jukebox':
            print(f'--- extracting jukebox for {f} with {layers_per} layers at a time ---', file=logfile_handle)
            # note that layers are 1-indexed in jukebox
            # so let's 0-idx and then add 1 when feeding into jukebox fn
            layer_gen = (list(range(l, min(um.model_num_layers['jukebox'], l + layers_per))) for l in range(0,um.model_num_layers['jukebox'], layers_per))
            has_last_layer = False
            if memmap == False:
                np_shape = um.get_embedding_shape(cur_act_type)
                np_arr = np.zeros(np_shape)
            if layer_num > 0:
                # 0-idx from 1-idxed argt
                layer_gen = ([l-1] for l in [layer_num])
            for layer_arr in layer_gen:
                # 1-idx for passing into fn
                j_idx = [l+1 for l in layer_arr]
                has_last_layer = um.model_num_layers['jukebox'] in j_idx
                print(f'extracting layers {j_idx}', file=logfile_handle)
                rep_arr = get_jukebox_layer_embeddings(fpath=fpath, audio = audio, layers=j_idx)
                if memmap == True:
                    emb_file[layer_arr,:] = rep_arr
                    emb_file.flush()
                else:
                    np_arr[layer_arr,:] = rep_arr
                    # should be the last layer to save
                    if has_last_layer == True:
                        um.save_npy(np_arr, out_fname, cur_act_type, acts_folder = acts_folder, dataset=cur_dataset)
        else:
            audio_ipt = fdict['audio']
            if model_longhand == "musicgen-encoder":
                print(f'--- extracting musicgen-encoder for {f} ---', file=logfile_handle)

                rep_arr = get_musicgen_encoder_embeddings(model, proc, audio_ipt, meanpool = True, model_sr = model_sr, device=device)
                if memmap == True:
                    emb_file[:,:] = rep_arr
                    emb_file.flush()
                else:
                    um.save_npy(rep_arr, out_fname, cur_act_type, acts_folder = acts_folder, dataset=cur_dataset)
            else:

                print(f'--- extracting musicgen_lm for {f} ---', file=logfile_handle)
                rep_arr =  get_musicgen_lm_hidden_states(model, proc, audio_ipt, text="", meanpool = True, model_sr = model_sr, device=device)
                if memmap == True:
                    emb_file[:,:] = rep_arr
                    emb_file.flush()
                else:
                    um.save_npy(rep_arr, out_fname, cur_act_type, acts_folder = acts_folder, dataset=cur_dataset)
        fname = fdict['fname']
        print(f'{fname},1', file=recfile_handle)



# note that these are 32bit
def get_baselines(cur_act_type, cur_dataset, normalize = True, dur = 4., logfile_handle=None, recfile_handle = None, memmap = True, pickup = False):
    #cur_model_type = um.get_model_type(cur_act_type)
    #model_sr = um.model_sr[cur_model_type]
    #model_longhand = um.model_longhand[cur_act_type]
   
    shape_path = um.get_baseline_shape_csv_path(cur_dataset)
    shape_file  = open(shape_path, 'a')
    is_shape_file_empty = um.is_file_empty(shape_file)
    if is_shape_file_empty == True:
        print('emb_type,fname,shape', file=shape_file)
    using_hf = cur_dataset in um.hf_datasets
    # musicgen stuff
    device = 'cpu'
    num_layers = None
    proc = None
    model = None
    text = ""
    wav_path = os.path.join(um.by_projpath('wav'), cur_dataset)
    cur_pathlist = None

    out_ext = 'dat'
    if memmap == False:
        out_ext = 'npy'

    if using_hf == True:
        cur_pathlist = uhf.load_syntheory_train_dataset(cur_dataset)
    else:
        cur_pathlist = um.path_list(wav_path)
    
    # original was 22050 but why not double
    sr = 44100

    existing_name_set = None
    if pickup == True:
        _file_dir = um.get_model_act_path(cur_act_type, acts_folder = acts_folder, dataset=cur_dataset, return_relative = False, make_dir = False)
        existing_files = um.remove_latest_file(_file_dir, is_relative = False)
        existing_name_set = set([um.get_basename(_f, with_ext = False) for _f in existing_files])

    for fidx,f in enumerate(cur_pathlist):
        if pickup == True:
            cur_name = um.get_basename(f, with_ext = False)
            if cur_name in existing_name_set:
                continue

        fdict = path_handler(f, model_sr = sr, wav_path = wav_path, normalize = normalize, dur = dur,model_type = 'baseline', using_hf = using_hf, logfile_handle=logfile_handle, out_ext = out_ext)
        #outpath = os.path.join(out_dir, outname)
        out_fname = fdict['out_fname']
        fpath = fdict['in_fpath']
        audio = fdict['audio']
        # store by 'baseline_mel','baseline_chroma','baseline_mfcc','baseline_concat'
        acts_to_get = set([])
        if cur_act_type == 'baseline_all':
            acts_to_get = acts_to_get.union(um.baseline_names)
        elif cur_act_type in um.baseline_names:
            acts_to_get = acts_to_get.add(cur_act_type)
        for _act in acts_to_get:
            ft_vec = get_baseline_features(audio, sr=sr, feat_type=_act)
            ft_shape = ft_vec.shape
            ft_shape_str = um.get_shape_string(ft_shape, joiner='|')
            if memmap == True:
                emb_file = um.get_embedding_file(_act, acts_folder=acts_folder, dataset=cur_dataset, fname=out_fname, use_64bit = False, write=True, use_shape = ft_shape)
                emb_file[:,:] = ft_vec
                emb_file.flush()
            else:
                um.save_npy(ft_vec, out_fname, _act, acts_folder = acts_folder, dataset=cur_dataset)

            print(f'{_act},{out_fname},{ft_shape_str}', file=shape_file)
        fname = fdict['fname']
        print(f'{fname},1', file=recfile_handle)

    shape_file.close()

def get_print_name(dataset, act_type, is_csv = False, normalize = True, timestamp = 0):
    base_fname = f'{dataset}_{act_type}-{timestamp}'
    if normalize == True:
        base_fname = f'{dataset}_{act_type}_norm-{timestamp}'
    ret = None
    if is_csv == False:
        ret = f'{base_fname}.log'
    else:
        ret = f'{base_fname}.csv'
    return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ub", "--use_64bit", type=strtobool, default=True, help="use 64-bit")
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-at", "--activation_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox /baseline_{mel/chroma/mfcc/concat/all}")
    parser.add_argument("-lp", "--layers_per", type=int, default=4, help="layers per loop if doing all layers (for jukebox)")
    parser.add_argument("-l", "--layer_num", type=int, default=-1, help="1-indexed layer num (all if < 0, for jukebox)")
    parser.add_argument("-n", "--normalize", type=strtobool, default=True, help="normalize audio")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="save as memmap, else save as npy")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="debug mode")
    parser.add_argument("-p", "--pickup", type=strtobool, default=False, help="pickup where script left off")


    args = parser.parse_args()
    use_64bit = args.use_64bit
    lnum = args.layer_num
    lper = args.layers_per
    memmap = args.memmap
    normalize = args.normalize
    act_type = args.activation_type
    dataset = args.dataset
    debug = args.debug
    pickup = args.pickup
    # exit if not a "real" dataset
    logdir = um.by_projpath(subpath='log', make_dir = True)
    timestamp = int(time.time() * 1000)

    # miscellaneous logs
    log_fname = get_print_name(dataset, act_type, is_csv = False, normalize = normalize, timestamp = timestamp)
    rec_fname = get_print_name(dataset, act_type, is_csv = True, normalize = normalize, timestamp = timestamp)
    log_fpath = os.path.join(logdir, log_fname)
    rec_fpath = os.path.join(logdir, rec_fname)
    if debug == True:
        exit()
    if (dataset in um.all_datasets) == False:
        sys.exit('not a dataset')
    else:
        lf = open(log_fpath, 'a')
        rf = open(rec_fpath, 'w')
        print(f'=== running extraction for {dataset} with {act_type} at {timestamp} ===', file=lf)
        if 'baseline' in act_type:
            get_baselines(act_type, dataset, normalize=normalize, dur = dur, logfile_handle = lf, recfile_handle =rf, memmap = memmap, pickup = pickup)
        else:
            get_embeddings(act_type, dataset, layers_per = lper, layer_num = lnum, normalize = normalize, dur = dur, use_64bit = use_64bit, logfile_handle=lf, recfile_handle=rf, memmap = memmap, pickup = pickup)
        lf.close()
        rf.close()
