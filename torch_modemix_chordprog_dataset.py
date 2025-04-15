import torch
import torch.utils.data as TUD
import os
import polars as pl
from sklearn import preprocessing as SKP
import torch.nn.functional as NF
import util as UM
import chords as CH
import chordprog as CHP
import numpy as np
from sklearn.model_selection import train_test_split
import utils_probing as UP

# classify by subcategory = classify by individual progression (else classify by modemix or not)
# inv, if >= 0, filter by this inversion
class ModemixChordprogData(TUD.Dataset):
    def __init__(self, cur_df, embedding_type = 'mg_small_h', device='cpu', layer_idx=-1, classify_by_subcategory = True, is_64bit = True, save_ext = 'dat'):
        self.device = device
        self.is_64bit = is_64bit
        self.embedding_type = embedding_type
        

        self.classify_by_subcategory = classify_by_subcategory
        self.data = cur_df.with_columns(

                sub_prog_idx=pl.col('sub_prog').replace_strict(CHP.subp_to_idx).cast(int),
                is_modemix_idx=pl.col('is_modemix').replace_strict(CHP.imm_to_idx).cast(int))



        self.all_subprog = self.data.select(['sub_prog']).to_numpy().flatten()
        self.all_subprog_idx = self.data.select(['sub_prog_idx']).to_numpy().flatten()

        self.all_imm = self.data.select(['is_modemix']).to_numpy().flatten()
        self.all_imm_idx = self.data.select(['is_modemix_idx']).to_numpy().flatten()
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
        if self.classify_by_subcategory == True:
            cur_truth = cur_row[self.coldict['sub_prog_idx']]
        else:
            cur_truth = cur_row[self.coldict['is_modemix_idx']]
        cur_arr = None
        if self.save_ext == 'dat':
            cur_fname = f'{cur_name}.dat'
            cur_arr =  UM.embedding_file_to_torch(self.embedding_type, acts_folder = 'acts', dataset='modemix_chordprog', fname=cur_fname, layer_idx = self.layer_idx, device = self.device, use_64bit = self.is_64bit)
        else:
            cur_fname = f'{cur_name}.npy'
            cur_arr = UM.npy_to_torch(self.embedding_type, acts_folder = 'acts', dataset='modemix_chordprog', fname=cur_fname, layer_idx = self.layer_idx, use_64bit = self.is_64bit, device = self.device)
        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        return cur_arr, cur_truth


