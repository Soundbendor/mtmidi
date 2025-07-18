import torch
import torch.utils.data as TUD
import os
import polars as pl
from sklearn import preprocessing as SKP
import torch.nn.functional as NF
import util as UM
import polyrhythms as PL
import numpy as np
from sklearn.model_selection import train_test_split
import utils_probing as UP
import util_data as UD

# exclude_polys should be in pstr format
class PolyrhythmsData(TUD.Dataset):
    def __init__(self, cur_df, embedding_type = 'mg_small_h', device='cpu', classification = True, classdict={}, norm_labels = True, layer_idx=-1, is_64bit = True, is_memmap = True):
        self.device = device
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
        

        # filter out exclude_polys and exclude_offset_lvls (keep given matching both nonexcluded) 
        # also sort by norm_ratio ascending
        # also map the 'poly' to label indices

        self.data = cur_df.sort('norm_ratio', descending=False)
        self.all_pstr = self.data.select(['poly']).to_numpy().flatten()
        self.all_offset_lvls = self.data.select(['offset_lvl']).to_numpy().flatten()
        self.total_num = self.data['name'].count()
        #self.data_folder = os.path.join(UM.by_projpath('acts'), 'polyrhythms', embedding_type)
        self.coldict = {x:i for (i,x) in enumerate(self.data.columns)}
        self.classification = classification
        self.norm_labels = norm_labels
        self.classdict = classdict
        self.is_memmap = is_memmap
    def __len__(self):
        return self.data['name'].count()

    def set_layer_idx(self, idx):
        self.layer_idx = idx

    def __getitem__(self, idx):
        cur_truth = None
        cur_row = self.data.row(idx)
        cur_name = cur_row[self.coldict['name']]
        cur_reg = None
        cur_label = cur_row[self.coldict['poly']]
        cur_truth = self.classdict[cur_label]
        if self.classification == False:
            if self.norm_labels == True:
                cur_reg = cur_row[self.coldict['norm_ratio']]
            else:
                cur_reg = cur_row[self.coldict['ratio']]
        cur_arr = UD.get_data_vec_at_idx(cur_name, self.layer_idx, self.embedding_type, is_memmap = self.is_memmap, acts_folder = 'acts', dataset = 'polyrhythms', to_torch = True, use_64bit = self.is_64bit, device = self.device)
        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        if self.classification == True:
            return cur_arr, cur_truth
        else:
            return cur_arr, cur_reg, cur_truth


