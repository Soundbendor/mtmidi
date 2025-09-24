import os,sys,time,argparse
import tomllib
import copy
from distutils.util import strtobool
from functools import partial
import optuna
import torch
import pickle
import numpy as np
from torch import nn
import torch.utils.data as TUD
import util as UM
import utils_probing as UP
import util_data as UD
import optuna_utils as OU
import torch_nep as TN
import polyrhythms as PL
import dynamics as DYN
import chords7 as CH7
import hf_chords as HFC
import hf_timesig as HTS
import hf_simpleprog as HFSP
import tempi as TP
import chords as CHS
import chordprog as CHP
import chord7prog as CSP
import util_db as UB
from torch_scalers import StandardScaler as TST
from torch_polyrhythms_dataset import PolyrhythmsData
from torch_dynamics_dataset import DynamicsData
from torch_modemix_chordprog_dataset import ModemixChordprogData
from torch_secondary_dominant_dataset import SecondaryDominantData
from torch_chords7_dataset import Chords7Data

from hf_tempi_dataset import STHFTempiData
from hf_chords_dataset import STHFChordsData
from hf_timesig_dataset import STHFTimeSignaturesData
from hf_simpleprog_dataset import STHFSimpleProgressionsData


_classify_by_subcategory = False
tomlfile_str = ''
use_folds = True
train_on_middle = False
train_pct = 0.7
test_subpct = 0.5
seed = 5
is_split_debug = False
device = 'cpu'
is_64bit = False
emb_type = 'mg_audio'
layer_idx = 0
is_memmap = True
is_classification = True

split_debug_name = ''
for cur_dsname in ['polyrhythms', 'secondary_dominant', 'chords7', 'dynamics', 'modemix_chordprog']: 
    if cur_dsname == 'polyrhythms':
        tomlfile_str = 'poly_ex3'
    else:
        tomlfile_str = ''
    datadict  = UD.load_data_dict(cur_dsname, classify_by_subcategory = _classify_by_subcategory, tomlfile_str = tomlfile_str, use_folds = use_folds)
    out_dim = datadict['num_classes']
    cur_df = datadict['df']
    pl_classdict = datadict['pl_classdict']
    #### load dataset(s)
    if cur_dsname == "polyrhythms":

        cur_ds = PolyrhythmsData(cur_df, embedding_type = emb_type, device=device, classification = is_classification, classdict = pl_classdict, norm_labels = True, layer_idx=layer_idx, is_64bit = is_64bit, is_memmap = is_memmap)
    elif cur_dsname == 'tempos':
        cur_ds = STHFTempiData(cur_df, embedding_type= emb_type, device=device, norm_labels = True, layer_idx= layer_idx, class_binsize = TEMPOS_CLASS_BINSIZE, num_classes = TP.num_classes, bpm_class_mapper = TP.bpm_class_mapper, is_64bit = is_64bit, is_memmap = is_memmap)
    elif cur_dsname == 'dynamics':
        cur_ds = DynamicsData(cur_df, embedding_type = emb_type, device=device, layer_idx=layer_idx, classify_by_subcategory = _classify_by_subcategory, is_64bit = is_64bit, is_memmap = is_memmap)
    elif cur_dsname == 'chords7':
        cur_ds = Chords7Data(cur_df, embedding_type = emb_type, device=device, layer_idx=layer_idx, is_64bit = is_64bit,is_memmap = is_memmap)
    elif cur_dsname == 'chords':
        cur_ds = STHFChordsData(cur_df, embedding_type = emb_type, device=device, layer_idx=layer_idx, is_64bit = is_64bit,is_memmap = is_memmap)
    elif cur_dsname == 'time_signatures':
        cur_ds = STHFTimeSignaturesData(cur_df, embedding_type = emb_type, device=device, layer_idx=layer_idx, is_64bit = is_64bit,is_memmap = is_memmap)
    elif cur_dsname == 'simple_progressions':
        cur_ds = STHFSimpleProgressionsData(cur_df, embedding_type = emb_type, device=device, layer_idx=layer_idx, classify_by_subcategory = _classify_by_subcategory, is_64bit = is_64bit, is_memmap = is_memmap)
        
    elif cur_dsname == 'modemix_chordprog':
        cur_ds = ModemixChordprogData(cur_df, embedding_type = emb_type, device=device, layer_idx=layer_idx, classify_by_subcategory = _classify_by_subcategory, is_64bit = is_64bit, is_memmap = is_memmap)
    elif cur_dsname == 'secondary_dominant':
        cur_ds = SecondaryDominantData(cur_df, embedding_type = emb_type, device=device, layer_idx=layer_idx, classify_by_subcategory = _classify_by_subcategory, is_64bit = is_64bit, is_memmap = is_memmap)
 
    label_arr = datadict['label_arr']
    cur_subset = UP.torch_get_train_test_subsets(cur_ds, cur_df, label_arr, train_on_middle = train_on_middle, train_pct = train_pct, test_subpct = test_subpct,seed = seed, debug=is_split_debug, debug_name=split_debug_name, use_folds = use_folds)
    tr_idxs = cur_subset['train_idx']
    v_idxs = cur_subset['valid_idx']
    te_idxs = cur_subset['test_idx']

    overlaps = False
    t_te = set(tr_idxs).intersection(set(te_idxs))
    te_v = set(te_idxs).intersection(set(v_idxs))
    t_v = set(tr_idxs).intersection(set(v_idxs))
    if len(t_te) > 0 or len(te_v) > 0 or len(t_v) > 0:
        overlaps = True
    if overlaps == True:
        print(f'overlap with {cur_dsname}')
        break
