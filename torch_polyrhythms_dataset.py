import torch
import os
import polars as pl
from sklearn import preprocessing as SKP
import torch.nn.functional as NF
import util as UM
import polyrhythms as PL
import numpy as np
from sklearn.model_selection import train_test_split

# exclude_polys should be in pstr format
class PolyrhythmsData(torch.utils.data.Dataset):
    def __init__(self, embedding_type = 'mg_small_h', device='cpu', classification = True, exclude_polys = [], exclude_offset_lvls = [], norm_labels = True, layer_idx=-1):
        self.device = device
        self.embedding_type = embedding_type
        cur_data = pl.scan_csv(csvfile).collect()
        self.all_pstr = cur_data.select(['poly']).to_numpy().flatten()
        self.all_offset_lvls = cur_data.select(['offset_lvl']).to_numpy().flatten()
        # filter out exclude_polys and exclude_offset_lvls
        # also sort by norm_ratio ascending
        # also map the 'poly' to label indices
        self.data = cur_data.filter((pl.col('poly').is_in(np.setdiff1d(self.all_pstr, exclude_polys)) & pl.col('offset_lvl').is_in(np.setdiff1d(self.all_offset_lvls, exclude_offset_lvls)))).sort('norm_ratio', descending=False).with_columns(pl.col('poly').map_elements(PL.get_idx_from_polystr, return_dtype=int).alias('label_idx'))
        self.total_num = self.data['name'].count()
        self.data_folder = os.path.join(UM.by_projpath('acts'), 'polyrhythms', embedding_type)
        self.classification = classification
        self.norm_labels = norm_labels
        self.classdict = PL.polystr_to_idx
    def __len__(self):
        return self.data['name'].count()

    def __getitem__(self, idx):
        cur_truth = None
        cur_name = self.data['name'][idx]
        cur_reg = None
        cur_label = self.data['poly'][idx]
        cur_truth = self.classdict[cur_label]
        if self.classification == False:
            if self.norm_labels == True:
                cur_reg = self.data['norm_ratio'][idx]
            else:
                cur_reg = self.data['ratio'][idx]
        fpath = os.path.join(self.data_folder, f'{cur_name}.pt')
        cur_arr = None
        if self.layer_idx < 0:
            cur_arr = torch.load(fpath, map_location=torch.device(self.device))
        else:
            cur_arr = torch.load(fpath, map_location=torch.device(self.device))[self.layer_idx,:]

        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        if self.classification == True:
            return cur_arr, cur_truth
        else:
            return cur_arr, cur_reg, cur_truth


# train_pct refers to entire dataset, test_subpct refers to length after split
def get_train_valid_test_idx(poly_dataset, train_on_middle = True, train_pct = 0.7, test_subpct = 0.5, seed = 5):
    test_valid_pct = 1. - train_pct
    valid_pct = (1. - test_subpct) * test_pct
    test_pct = test_subpct * test_pct
    train_idx = None
    test_valid_idx = None
    total_num = poly_dataset.total_num
    all_idx = np.arange(0, total_num)
    labels = poly_dataset['label_idx'].to_numpy()
    if train_on_middle == False:
        _train_idx, _test_valid_idx = train_test_split(all_idx, random_state = seed, shuffle = True, stratify=labels)
        train_idx = np.array(train_idx)
        test_valid_idx = np.array(test_valid_idx)
    else:
        # getting start index of train, starting with valid_pct arbitrarily
        train_start = int(valid_pct * total_num)
        train_end = int((1. - test_pct) * total_num)
        train_idx =  all_idx[train_start:train_end]
        test_valid_idx = np.concatenate((all_idx[:train_start],all_idx[train_end:]))
    leftover_labels = labels[test_valid_idx]
    # returns indices of our index lists so we have to convert to regular indices
    _test_idx, _valid_idx = train_test_split(test_valid_idx, random_state = seed, shuffle= True, stratify=leftover_labels)
    test_idx = test_valid_idx[_test_idx]
    valid_idx = test_valid_idx[_valid_idx]
    return train_idx, valid_idx, test_idx



        
