import torch
import torch.utils.data as TUD
import os
import polars as pl
from sklearn import preprocessing as SKP
import torch.nn.functional as NF
import util as UM
import chords
import numpy as np
from sklearn.model_selection import train_test_split
import utils_probing as UP

class ChordsData(TUD.Dataset):
    def __init__(self, embedding_type = 'mg_small_h', device='cpu', exclude = [],layer_idx=-1, is_64bit = True):
        self.device = device
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
        csvfile = os.path.join(UM.by_projpath('csv', make_dir = False), 'chords.csv')
        cur_data = pl.scan_csv(csvfile).collect()
        

        self.all_quality = cur_data.select(['quality']).to_numpy().flatten()
        self.all_quality_idx = cur_data.select(['quality_idx']).to_numpy().flatten()


        cur_data = UP.exclude_col_vals_in_data(cur_data, exclude)
        self.data = cur_data

        self.total_num = self.data['name'].count()
        self.coldict = {x:i for (i,x) in enumerate(self.data.columns)}

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
        cur_truth = cur_row[self.coldict['quality_idx']]
        cur_arr =  UM.embedding_file_to_torch(self.embedding_type, acts_folder = 'acts', dataset='chords', fname=cur_fname, layer_idx = self.layer_idx, device = self.device, use_64bit = self.is_64bit)

        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        return cur_arr, cur_truth


