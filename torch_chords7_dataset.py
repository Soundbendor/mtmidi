import torch
import torch.utils.data as TUD
import os
import polars as pl
from sklearn import preprocessing as SKP
import torch.nn.functional as NF
import util as UM
import chords7 as CH7
import numpy as np
from sklearn.model_selection import train_test_split
import utils_probing as UP
import util_data as UD

class Chords7Data(TUD.Dataset):
    def __init__(self, cur_df, embedding_type = 'mg_small_h', device='cpu', layer_idx=-1, is_64bit = True, save_ext = 'dat'):
        self.device = device
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
        



        self.data = cur_df

        self.all_quality_idx = self.data.select(['quality_idx']).to_numpy().flatten()
        self.total_num = self.data['name'].count()
        self.coldict = {x:i for (i,x) in enumerate(self.data.columns)}
        self.layer_idx = layer_idx
        self.save_ext = save_ext
    def __len__(self):
        return self.data['name'].count()

    def set_layer_idx(self, idx):
        self.layer_idx = idx

    def set_classify_by_subcategory(self, to_set):
        self.classify_by_subcategory = to_set == True

    def __getitem__(self, idx):
        cur_truth = None
        cur_row = self.data.row(idx)
        cur_name = cur_row[self.coldict['name']]
        cur_truth = cur_row[self.coldict['quality_idx']]
        cur_arr = UD.get_data_vec_at_idx(cur_name, self.layer_idx, self.embedding_type, save_ext = self.save_ext, acts_folder = 'acts', dataset = 'chords7', to_torch = True, use_64bit = self.is_64bit, device = self.device)
        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        return cur_arr, cur_truth


