import torch
import torch.utils.data as TUD
import os
import polars as pl
from sklearn import preprocessing as SKP
import torch.nn.functional as NF
import util as UM
import chords7 as CHS
import chord7prog as CSP
import numpy as np
from sklearn.model_selection import train_test_split
import utils_probing as UP
import util_data as UD

# classify by subcategory = classify by individual progression (else classify by none,secondary,tritone)
# inv, if >= 0, filter by this inversion
class SecondaryDominantData(TUD.Dataset):
    def __init__(self, cur_df, embedding_type = 'mg_small_h', device='cpu', layer_idx=-1, classify_by_subcategory = True, is_64bit = True, is_memmap = True):
        self.device = device
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
        

        self.classify_by_subcategory = classify_by_subcategory
        self.data = cur_df.with_columns(

                sub_prog_idx=pl.col('sub_prog').replace_strict(CSP.subp_to_idx).cast(int),
                sub_type_idx=pl.col('sub_type').replace_strict(CSP.sub_type_to_idx).cast(int))



        self.all_subprog_idx = self.data.select(['sub_prog_idx']).to_numpy().flatten()

        self.all_subtypes_idx = self.data.select(['sub_type_idx']).to_numpy().flatten()
        self.total_num = self.data['name'].count()
        self.coldict = {x:i for (i,x) in enumerate(self.data.columns)}
        self.layer_idx = layer_idx
        self.is_memmap = is_memmap
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
        if self.classify_by_subcategory == True:
            cur_truth = cur_row[self.coldict['sub_prog_idx']]
        else:
            cur_truth = cur_row[self.coldict['sub_type_idx']]
        cur_arr = UD.get_data_vec_at_idx(cur_name, self.layer_idx, self.embedding_type, is_memmap = self.is_memmap, acts_folder = 'acts', dataset = 'secondary_dominant', to_torch = True, use_64bit = self.is_64bit, device = self.device)
        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        return cur_arr, cur_truth



