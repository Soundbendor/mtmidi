import jukemirlib as jml
import os
import torch

out_dir = 'jukebox_acts'
load_dir = 'wav_trim'
log = 'jml.log'
dsamp_rate = 22050
#layer_acts = [36,72]
layer_acts = [x for x in range(1,73)]
num_layers = len(layer_acts)
num_per = 3
if os.path.isfile(log):
    os.remove(log)
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)
for fidx,f in enumerate(os.listdir(load_dir)):
    #if fidx > 0: break

    fsplit = '.'.join(f.split('.')[:-1])
    outname = f'{fsplit}.pt'
    outpath = os.path.join(out_dir, outname)

    fpath = os.path.join(load_dir, f)
    audio = jml.load_audio(fpath)
    first_layer_done = False
    means = None
    with open(log, 'a') as wf:
        print(f'loading {fpath}', file=wf)
        for layer_start in range(0, num_layers, num_per):
            cur_layers = layer_acts[layer_start:layer_start+num_per]
            print(cur_layers, "-----")
            #if first_layer_done == True:break
            print(f'getting reps for layers {cur_layers}', file=wf)
            reps = jml.extract(audio, layers=cur_layers, downsample_target_rate=dsamp_rate, meanpool = True)
            for layer_idx in cur_layers:
                #print(layer_idx)
                if first_layer_done == False:
                    first_layer_done = True
                    means = torch.from_numpy(reps[layer_idx])
                else:
                    #pass
                    means = torch.vstack((means, torch.from_numpy(reps[layer_idx])))
        torch.save(means, outpath)
        print(means, file=wf)
        print(means.shape,file=wf)
