# https://github.com/brown-palm/syntheory/blob/4f222359e750ec55425c12809c1a0358b74fce49/probe/probes.py#L187

import torch
import os
import polars as pl
from sklearn import preprocessing as SKP
from sklearn.model_selection import train_test_split
import torch.nn.functional as NF
from datasets import load_dataset
#import numpy as np

class STHFTempi(torch.utils.data.Dataset):
    def __init__(self, csvfile=os.path.join('hf_csv', 'tempi.csv'), data_folder='jukebox_acts_38', set_type='train', device='cpu', norm_labels = True, layer_idx=-1):
        self.seed = 5
        self.set_type = set_type
        self.device = device
        cur_data = pl.scan_csv(csvfile).collect()
        cur_data = cur_data.sort('bpm', descending=False)
        self.min_bpm = cur_data['bpm'].min()
        self.max_bpm = cur_data['bpm'].max()
        self.bpm_range = self.max_bpm - self.min_bpm
        #self.scalefunc = np.vectorize(lambda x: (x - self.min_bpm)/self.bpm_range)
        #cur_data = cur_data.with_columns(norm_bpm=self.scalefunc(cur_data['bpm']))
        total_sz = cur_data['path'].count()
        test_valid_prop = 0.15
        train_st_idx = int(total_sz * test_valid_prop)
        train_end_idx = int(total_sz * (1. - test_valid_prop))
        if set_type == 'train':
            self.data = cur_data[train_st_idx:train_end_idx]
        else:
            test_valid = pl.concat((cur_data[:train_st_idx], cur_data[train_end_idx:]))
            num_tv = test_valid['path'].count()
            seq = list(range(num_tv))
            # should be the same each time with a seed...
            tts = train_test_split(seq, test_size = 0.5, random_state = self.seed)
            self.test_idxs = tts[0]
            self.valid_idxs = tts[1]
            if set_type == "test":
                self.data = cur_data[self.test_idxs]
            else:
                self.data = cur_data[self.valid_idxs]

        self.norm_labels = norm_labels
        self.data_folder = data_folder
        self.layer_idx = layer_idx 
    def __len__(self):
        return self.data['path'].count()

    def __getitem__(self, idx):
        cur_truth = None
        cur_name = self.data['name'][idx]
        cur_reg = None
        cur_label = self.data['poly'][idx]
        cur_truth = self.classdict[cur_label]
        if self.norm_labels == True:
            cur_reg = self.data['norm_bpm'][idx]
        else:
            cur_reg = self.data['bpm'][idx]
        fpath = os.path.join(self.data_folder, f'{cur_name}.pt')
        cur_arr = None
        if self.layer_idx < 0:
            cur_arr = torch.load(fpath, map_location=torch.device(self.device))
        else:
            cur_arr = torch.load(fpath, map_location=torch.device(self.device))[self.layer_idx]

        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        return cur_arr, cur_reg, cur_truth



