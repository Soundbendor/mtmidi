# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py

import jukemirlib as jml
import os
import torch
import librosa
import util as um
out_dir = um.by_projpath(os.path.join('acts', 'jukebox_acts_36'), make_dir = True)
load_dir = um.by_projpath('wav')
log = um.by_projpath(os.path.join('log', 'jml.log'))
#dsamp_rate = 22050
dsamp_rate = 15
layer_act = 36
dur = 4.0
#layer_acts = [x for x in range(1,73)]
if os.path.isfile(log):
    os.remove(log)
with open(log, 'a') as lf:
    for fidx,f in enumerate(os.listdir(load_dir)):
        #if fidx > 0: break
        
        fsplit = '.'.join(f.split('.')[:-1])
        outname = f'{fsplit}.pt'
        outpath = os.path.join(out_dir, outname)

        fpath = os.path.join(load_dir, f)
        print(f'loading {fpath}', file=lf)
        reps = jml.extract(fpath = fpath, layers=[layer_act], duration= dur, downsample_method=None, downsample_target_rate=dsamp_rate, meanpool = True)
        torch.save(reps[layer_act], outpath)
        jml.lib.empty_cache()
                   
