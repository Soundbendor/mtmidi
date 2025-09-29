# https://github.com/brown-palm/syntheory/blob/4f222359e750ec55425c12809c1a0358b74fce49/probe/probes.py#L187

import torch
import os
import polars as pl
from sklearn import preprocessing as SKP
import sklearn.utils as SU
from sklearn.model_selection import train_test_split
import torch.nn.functional as NF
from datasets import load_dataset
import numpy as np
import util as UM
import util_data as UD


# class_binsize = bpms within int(bpm/10) go within this bin
class STHFTempiData(torch.utils.data.Dataset):
    def __init__(self, cur_df, embedding_type='mg_small_h', device='cpu', norm_labels = True, layer_idx=-1, class_binsize = 10, is_64bit = True, num_classes = -1, bpm_class_mapper = None, is_memmap = True, on_share = False):
        self.seed = 5
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
        self.device = device
        self.on_share = on_share
        self.class_binsize = class_binsize
        cur_df = cur_df.sort('bpm', descending=False)
        self.min_bpm = cur_df['bpm'].min()
        self.max_bpm = cur_df['bpm'].max()
        self.bpm_range = self.max_bpm - self.min_bpm
        self.num_classes = num_classes
        self.bpm_class_mapper = bpm_class_mapper
        cur_df = cur_df.with_columns(pl.col('bpm').map_elements(self.bpm_class_mapper, return_dtype=int).alias('bpm_class'))
        self.data = cur_df
        # class dict with medians
        #self.scalefunc = np.vectorize(lambda x: (x - self.min_bpm)/self.bpm_range)
        #cur_data = cur_data.with_columns(norm_bpm=self.scalefunc(cur_data['bpm']))

        self.total_num = self.data['path'].count()
        self.coldict = {x:i for (i,x) in enumerate(self.data.columns)}
        self.norm_labels = norm_labels
        #self.data_folder = os.path.join(UM.by_projpath('acts'),'tempi', embedding_type)
        self.layer_idx = layer_idx 
        self.is_memmap = is_memmap
    def __len__(self):
        return self.data['path'].count()

    def set_layer_idx(self, idx):
        self.layer_idx = idx

    def __getitem__(self, idx):
        cur_truth = None
        cur_row = self.data.row(idx)
        cur_name = cur_row[self.coldict['name']]
        cur_reg = None
        #cur_truth = cur_row[self.coldict['bpm_class']]
        if self.norm_labels == True:
            cur_reg = cur_row[self.coldict['norm_bpm']]
        else:
            cur_reg = cur_row[self.coldict['bpm']]
        cur_arr = UD.get_data_vec_at_idx(cur_name, self.layer_idx, self.embedding_type, is_memmap = self.is_memmap, acts_folder = 'acts', dataset = 'tempos', to_torch = True, use_64bit = self.is_64bit, device = self.device, on_share = self.on_share)
        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        #return cur_arr, cur_reg, cur_truth
        return cur_arr, cur_reg



