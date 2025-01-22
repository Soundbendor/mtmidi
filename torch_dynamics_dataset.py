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

# exclude_dyn_pairs are given as 'dyn1-dyn2'
# either classify by category or subcategory
class DynamicsData(TUD.Dataset):
    def __init__(self, embedding_type = 'mg_small_h', device='cpu', exclude_dyn_pairs = [], exclude_dyn_categories= [], exclude_dyn_subcategories=[], norm_labels = True, layer_idx=-1, classify_by_subcategory = False, is_64bit = True):
        self.device = device
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
        csvfile = os.path.join(UM.by_projpath('csv', make_dir = False), 'dynamics.csv')
        cur_data = pl.scan_csv(csvfile).collect()
        self.all_dyn_pairs = cur_data.select(['dyn_pair']).to_numpy().flatten()
        self.all_dyn_categories = cur_data.select(['dyn_category']).to_numpy().flatten()
        self.all_dyn_subcategories = cur_data.select(['dyn_subcategory']).to_numpy().flatten()
        self.all_offset_lvls = cur_data.select(['offset_lvl']).to_numpy().flatten()
        # filter out dyn_pair, dyn_category, and dyn_subcategories (keep given matching all three nonexcluded)
        # map dyn_category to idx
        # map dyn_subcategory to idx
        self.data = cur_data.filter((pl.col('dyn_pair').is_in(np.setdiff1d(self.all_dyn_pairs, exclude_dyn_pairs)) & pl.col('dyn_category').is_in(np.setdiff1d(self.all_dyn_categories, exclude_dyn_categories)) & pl.col('dyn_subcategory').is_in(np.setdiff1d(self.all_dyn_subcategories, exclude_dyn_subcategories)) )).with_columns(pl.col('dyn_category').map_elements(DYN.get_category_idx, return_dtype=int).alias('category_idx')).with_columns(pl.col('dyn_subcategory').map_elements(DYN.get_subcategory_idx, return_dtype=int).alias('subcategory_idx'))

        self.total_num = self.data['name'].count()
        self.coldict = {x:i for (i,x) in enumerate(self.data.columns)}
        self.catdict = DYN.dyn_category_to_idx
        self.subcatdict = DYN.dyn_subcategory_to_idx
        self.classify_by_subcategory = classify_by_subcategory

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
        cur_fname = f'{cur_name}.dat'
        if self.classify_by_subcategory == True:
            cur_truth = cur_row[self.coldict['subcategory_idx']]
        else:
            cur_truth = cur_row[self.coldict['category_idx']]
        cur_arr =  UM.embedding_file_to_torch(self.embedding_type, acts_folder = 'acts', dataset='dynamics', fname=cur_fname, layer_idx = self.layer_idx, device = self.device, use_64bit = self.is_64bit)

        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        return cur_arr, cur_truth


