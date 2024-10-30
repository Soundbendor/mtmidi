import jukemirlib as jml
import os
import torch

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
    #if fidx > 0: break

    fsplit = '.'.join(f.split('.')[:-1])
    outname = f'{fsplit}.pt'
    outpath = os.path.join(out_dir, outname)

    fpath = os.path.join(load_dir, f)
    print(f'loading {fpath}')
    audio = jml.load_audio(fpath)
    first_layer_done = False
    means = None
    with open(log, 'a') as wf:
        for layer_act in layer_acts:
            print(f'getting reps for layer {layer_act}', file=wf)
            reps = jml.extract(audio, layers=[layer_act], downsample_target_rate=dsamp_rate, meanpool = True)
            if first_layer_done == False:
                first_layer_done = True
                means = torch.from_numpy(reps[layer_act])
            else:
                means = torch.vstack((means, torch.from_numpy(reps[layer_act])))
        torch.save(means, outpath)
        #print(means, file=wf)
        #print(means.shape,file=wf)

