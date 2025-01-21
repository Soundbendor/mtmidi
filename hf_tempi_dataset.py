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


# class_binsize = bpms within int(bpm/10) go within this bin
class STHFTempiData(torch.utils.data.Dataset):
    def __init__(self, embedding_type='mg_small_h', device='cpu', norm_labels = True, layer_idx=-1, class_binsize = 10, is_64bit = True, num_classes = -1, bpm_class_mapper = None):
        self.seed = 5
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
        self.device = device
        self.class_binsize = class_binsize
        csvfile = os.path.join(UM.by_projpath('hf_csv', make_dir = False), 'tempi.csv')
        cur_data = pl.scan_csv(csvfile).collect()
        cur_data = cur_data.sort('bpm', descending=False)
        self.min_bpm = cur_data['bpm'].min()
        self.max_bpm = cur_data['bpm'].max()
        self.bpm_range = self.max_bpm - self.min_bpm
        self.num_classes = num_classes
        self.bpm_class_mapper = bpm_class_mapper
        cur_data = cur_data.with_columns(bpm_class=self.bpm_class_mapper(cur_data['bpm'])).cast({'bpm_class': int})
        self.all_classes = cur_data['bpm_class'].to_numpy().flatten()
        self.data = cur_data
        # class dict with medians
        #self.scalefunc = np.vectorize(lambda x: (x - self.min_bpm)/self.bpm_range)
        #cur_data = cur_data.with_columns(norm_bpm=self.scalefunc(cur_data['bpm']))

        self.total_num = cur_data['path'].count()
        self.norm_labels = norm_labels
        self.data_folder = os.path.join(UM.by_projpath('acts'),'tempi', embedding_type)
        self.layer_idx = layer_idx 
    def __len__(self):
        return self.data['path'].count()

    def set_layer_idx(self, idx):
        self.layer_idx = idx

    def __getitem__(self, idx):
        cur_truth = None
        cur_name = self.data['name'][idx]
        cur_reg = None
        cur_truth = self.data['bpm_class'][idx]
        if self.norm_labels == True:
            cur_reg = self.data['norm_bpm'][idx]
        else:
            cur_reg = self.data['bpm'][idx]

        cur_arr =  UM.embedding_file_to_torch(self.embedding_type, acts_folder = 'acts', dataset='polyrhythms', fname=cur_name, write = False, layer_idx = self.layer_idx, device = self.device, use_64bit = self.is_64bit)
        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        return cur_arr, cur_reg, cur_truth



