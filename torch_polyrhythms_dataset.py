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

# exclude_polys should be in pstr format
class PolyrhythmsData(TUD.Dataset):
    def __init__(self, embedding_type = 'mg_small_h', device='cpu', classification = True, exclude_polys = [], exclude_offset_lvls = [], norm_labels = True, layer_idx=-1, is_64bit = True):
        self.device = device
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
        csvfile = os.path.join(UM.by_projpath('csv', make_dir = False), 'polyrhythms.csv')
        cur_data = pl.scan_csv(csvfile).collect()
        self.all_pstr = cur_data.select(['poly']).to_numpy().flatten()
        self.all_offset_lvls = cur_data.select(['offset_lvl']).to_numpy().flatten()
        # filter out exclude_polys and exclude_offset_lvls (keep given matching both nonexcluded) 
        # also sort by norm_ratio ascending
        # also map the 'poly' to label indices
        self.data = cur_data.filter((pl.col('poly').is_in(np.setdiff1d(self.all_pstr, exclude_polys)) & pl.col('offset_lvl').is_in(np.setdiff1d(self.all_offset_lvls, exclude_offset_lvls)))).sort('norm_ratio', descending=False).with_columns(pl.col('poly').map_elements(PL.get_idx_from_polystr, return_dtype=int).alias('label_idx'))
        self.total_num = self.data['name'].count()
        #self.data_folder = os.path.join(UM.by_projpath('acts'), 'polyrhythms', embedding_type)
        self.coldict = {x:i for (i,x) in enumerate(self.data.columns)}
        self.classification = classification
        self.norm_labels = norm_labels
        self.classdict = PL.polystr_to_idx
    def __len__(self):
        return self.data['name'].count()

    def set_layer_idx(self, idx):
        self.layer_idx = idx

    def __getitem__(self, idx):
        cur_truth = None
        cur_row = self.data.row(idx)
        cur_name = cur_row[self.coldict['name']]
        cur_fname = f'{cur_name}.dat'
        cur_reg = None
        cur_label = cur_row[self.coldict['poly']]
        cur_truth = self.classdict[cur_label]
        if self.classification == False:
            if self.norm_labels == True:
                cur_reg = cur_row[self.coldict['norm_ratio']]
            else:
                cur_reg = cur_row[self.coldict['ratio']]
        cur_arr =  UM.embedding_file_to_torch(self.embedding_type, acts_folder = 'acts', dataset='polyrhythms', fname=cur_fname, layer_idx = self.layer_idx, device = self.device, use_64bit = self.is_64bit)

        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        if self.classification == True:
            return cur_arr, cur_truth
        else:
            return cur_arr, cur_reg, cur_truth


