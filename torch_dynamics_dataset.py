import torch
import torch.utils.data as TUD
import os
import polars as pl
from sklearn import preprocessing as SKP
import torch.nn.functional as NF
import util as UM
import dynamics as DYN
import numpy as np
from sklearn.model_selection import train_test_split
import utils_probing as UP
import util_data as UD

# exclude_dyn_pairs are given as 'dyn1-dyn2'
# either classify by category or subcategory
# example categories to exclude: dyn_pair, dyn_category, dyn_subcategory, offset_lvl
class DynamicsData(TUD.Dataset):
    def __init__(self, cur_df, embedding_type = 'mg_small_h', device='cpu', layer_idx=-1, classify_by_subcategory = False, is_memmap = True, is_64bit = True):
        self.device = device
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
	
        
        # filter out dyn_pair, dyn_category, and dyn_subcategories (keep given matching all three nonexcluded)

        # map dyn_category to idx
        # map dyn_subcategory to idx
        self.data = cur_df.with_columns(

                category_idx=pl.col('dyn_category').replace_strict(DYN.dyn_category_to_idx).cast(int),
                subcategory_idx=pl.col('dyn_subcategory').replace_strict(DYN.dyn_subcategory_to_idx).cast(int))

        self.all_dyn_pairs = self.data.select(['dyn_pair']).to_numpy().flatten()
        self.all_offset_lvls = self.data.select(['offset_lvl']).to_numpy().flatten()

        self.total_num = self.data['name'].count()
        self.coldict = {x:i for (i,x) in enumerate(self.data.columns)}
        self.catdict = DYN.dyn_category_to_idx
        self.subcatdict = DYN.dyn_subcategory_to_idx
        self.classify_by_subcategory = classify_by_subcategory
        self.num_categories = DYN.num_categories
        self.num_subcategories = DYN.num_subcategories
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
            cur_truth = cur_row[self.coldict['subcategory_idx']]
        else:
            cur_truth = cur_row[self.coldict['category_idx']]
        cur_arr = UD.get_data_vec_at_idx(cur_name, self.layer_idx, self.embedding_type, is_memmap = self.is_memmap, acts_folder = 'acts', dataset = 'dynamics', to_torch = True, use_64bit = self.is_64bit, device = self.device)
        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        return cur_arr, cur_truth


