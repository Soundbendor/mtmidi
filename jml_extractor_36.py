import jukemirlib as jml
import os
import torch

out_dir = 'jukebox_acts_36'
load_dir = 'wav_trim'
log = 'jml.log'
dsamp_rate = 22050
layer_act = 36
#layer_acts = [x for x in range(1,73)]
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
    reps = jml.extract(audio, layers=[layer_act], downsample_target_rate=dsamp_rate, meanpool = True)
    torch.save(reps[layer_act], outpath)
