import numpy as np
import os
import util as UM

basedir = '/home/dxk/osu/mtmidi/acts2/'
acts_folder = 'acts'
cur_dataset = 'dynamics'
cdir = os.path.join(basedir, cur_dataset)
use_64bit = False

acts = os.listdir(cdir)
for actdir in acts:
    actpath = os.path.join(cdir,actdir)
    cur_num_layers = UM.get_embedding_num_layers(actdir)
    for f in os.listdir(actpath):
        fpath = os.path.join(actpath, f)
        cur_np = np.load(fpath)
        #cur_f = f
        #if f.dtype == 'float64':
            #print('64bit')

        print(actpath, cur_num_layers)
        new_fname = UM.ext_replace(f, 'dat')
        cur_f = cur_np.astype(np.float32)
        emb_file = UM.get_embedding_file(actdir, acts_folder=acts_folder, dataset=cur_dataset, fname=new_fname, use_64bit = use_64bit, write=True, use_shape = None)
        print(np.all(np.isclose(cur_f, emb_file)))
 
