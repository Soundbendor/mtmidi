import jukemirlib as jml
import os
import numpy as np

jml.setup_models(cache_dir='/nfs/guille/eecs_research/soundbendor/kwand/jukemirlib')
out_dir = 'jukebox_acts'
load_dir = 'wav_trim'
log = 'jml.log'
dsamp_rate = 22050
#layer_acts = [36,72]
layer_acts = [x for x in range(1,73)]
if os.path.isfile(log):
    os.remove(log)
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)
for fidx,f in enumerate(os.listdir(load_dir)):
    if fidx > 0: break
    
    fsplit = '.'.join(f.split('.')[:-1])
    outname = f'{fsplit}.npy'
    outpath = os.path.join(out_dir, outname)

    fpath = os.path.join(load_dir, f)
    print(f'loading {fpath}')
    audio = jml.load_audio(fpath)
    reps = jml.extract(audio, layers=layer_acts, downsample_target_rate=dsamp_rate, meanpool = True)
    first_layer_done = False
    means = None
    with open(log, 'a') as wf:
        for layer_idx, layer_act in reps.items():
            print(f'layer {layer_idx}', file=wf)
            print(reps[layer_idx], file=wf)
            print(reps[layer_idx].shape, file=wf)
            if first_layer_done == False:
                first_layer_done = True
                means = layer_act
            else:
                means = np.vstack((means, layer_act))
        print(means, file=wf)
        print(means.shape, file=wf)

                
