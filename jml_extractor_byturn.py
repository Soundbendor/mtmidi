import util as um
import jukemirlib as jml
import os
import torch

out_dir = um.by_projpath(os.path.join('acts', 'jukebox_acts'), make_dir = True)
load_dir = um.by_projpath('wav')
log = um.by_projpath(os.path.join('log', 'jml_bt.log'))
#dsamp_rate = 22050
dsamp_rate = 15
layer_acts = [x for x in range(1,73)]
num_layers = len(layer_acts)
num_per = 4
dur = 4.0
if os.path.isfile(log):
    os.remove(log)
for fidx,f in enumerate(os.listdir(load_dir)):
    #if fidx > 0: break
   
    fsplit = '.'.join(f.split('.')[:-1])
    outname = f'{fsplit}.pt'
    outpath = os.path.join(out_dir, outname)

    fpath = os.path.join(load_dir, f)
    first_layer_done = False
    means = None
    with open(log, 'a') as wf:
        print(f'loading {fpath}', file=wf)
        for layer_start in range(0, num_layers, num_per):
            cur_layers = layer_acts[layer_start:layer_start+num_per]
            print(cur_layers, "-----")
            #if first_layer_done == True:break
            print(f'getting reps for layers {cur_layers}', file=wf)
            reps = jml.extract(fpath = fpath, layers=cur_layers, duration= dur, downsample_method=None, downsample_target_rate=dsamp_rate, meanpool = True)
            for layer_idx in cur_layers:
                #print(layer_idx)
                if first_layer_done == False:
                    first_layer_done = True
                    means = torch.from_numpy(reps[layer_idx])
                else:
                    #pass
                    means = torch.vstack((means, torch.from_numpy(reps[layer_idx])))
            jml.lib.empty_cache()
        torch.save(means, outpath)
        print(means, file=wf)
        print(means.shape,file=wf)

             
