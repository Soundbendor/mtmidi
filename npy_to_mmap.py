import numpy as np
import os
import util as UM
import sys

basedir = '/home/dxk/osu/mtmidi/acts2/'
acts_folder = 'acts'
cur_dataset = sys.argv[1]
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
        print('cur_np', cur_np.dtype)
        cur_f = cur_np.astype(np.float32)
        shape = cur_f.shape
        emb_file = UM.get_embedding_file(actdir, acts_folder=acts_folder, dataset=cur_dataset, fname=new_fname, use_64bit = use_64bit, write=False, use_shape = shape)
        emb_file[:] = cur_f[:]
        print('cur_f', cur_f.dtype)

