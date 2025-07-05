import torch
import torch.utils.data as TUD
import os
import polars as pl
import hf_timesig as HTS
from sklearn import preprocessing as SKP
import torch.nn.functional as NF
import util as UM
import numpy as np
from sklearn.model_selection import train_test_split
import utils_probing as UP
import util_data as UD

class STHFTimeSignaturesData(TUD.Dataset):
    def __init__(self, cur_df, embedding_type = 'mg_small_h', device='cpu', layer_idx=-1, is_64bit = True, is_memmap = True):
        self.device = device
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
        



        self.data =  cur_df.with_columns(timesig_idx=pl.col('time_signature').replace_strict(HTS.timesig_to_idx).cast(int))


        self.all_timesig_idx = self.data.select(['timesig_idx']).to_numpy().flatten()
        self.total_num = self.data['name'].count()
        self.coldict = {x:i for (i,x) in enumerate(self.data.columns)}
        self.layer_idx = layer_idx
        self.is_memmap = is_memmap
    def __len__(self):
        return self.data['name'].count()

    def set_layer_idx(self, idx):
        self.layer_idx = idx


    def __getitem__(self, idx):
        cur_truth = None
        cur_row = self.data.row(idx)
        cur_name = cur_row[self.coldict['name']]
        cur_truth = cur_row[self.coldict['timesig_idx']]
        cur_arr = UD.get_data_vec_at_idx(cur_name, self.layer_idx, self.embedding_type, is_memmap = self.is_memmap, acts_folder = 'acts', dataset = 'time_signatures', to_torch = True, use_64bit = self.is_64bit, device = self.device)
        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        return cur_arr, cur_truth


