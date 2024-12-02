# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py

import jukemirlib as jml
import os
import torch
import librosa
import util as um
import argparse

jml.setup_models(cache_dir='/nfs/guille/eecs_research/soundbendor/kwand/jukemirlib')
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--layer_num", type=int, default=-1, help="layer num (all if < 0)")
parser.add_argument("-hf", "--hf_dataset", type=str, default="", help="use old hugging face dataset if passed")

args = parser.parse_args()
lnum = args.layer_num
hfds_str = args.hf_dataset

acts_dir = 'acts'
path_list = um.path_list('wav')

use_hf = len(hfds_str) > 0
if use_hf == True:
    path_list = uhf.load_syntheory_train_dataset(hfds_str)
    acts_dir = 'hf_acts'

out_folder = f'jukebox_acts_{lnum}'
if lnum <= 0:
    out_folder = f'jukebox_acts_all'
out_dir = um.by_projpath(os.path.join(acts_dir, out_folder), make_dir = True)
log = um.by_projpath(os.path.join('log', f'jml_{lnum}.log'))
#dsamp_rate = 22050
dsamp_rate = 15
layer_act = 36
dur = 4.0
#layer_acts = [x for x in range(1,73)]
if os.path.isfile(log):
    os.remove(log)
with open(log, 'a') as lf:
    for fidx,f in enumerate(path_list):
        #if fidx > 0: break
        
        outname = None
        if use_hf == False:
            print(f'loading {f}', file=lf)
            outname = um.ext_replace(f, new_ext="pt")
        else:
            print(f"loading {f['audio']['path']}", file=lf)
            outname = um.ext_replace(f['audio']['path'], new_ext="pt")
        outpath = os.path.join(out_dir, outname)

        if lnum > 0:
            reps = None
            if use_hf == False:
                reps = jml.extract(fpath = fpath, layers=[layer_act], duration= dur, downsample_method=None, downsample_target_rate=dsamp_rate, meanpool = True)
            else:

        torch.save(reps[layer_act], outpath)
        jml.lib.empty_cache()
                   
