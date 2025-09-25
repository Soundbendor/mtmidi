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
from sklearn.preprocessing import StandardScaler as KST
from torch_polyrhythms_dataset import PolyrhythmsData
from torch_dynamics_dataset import DynamicsData
from torch_modemix_chordprog_dataset import ModemixChordprogData
from torch_secondary_dominant_dataset import SecondaryDominantData
from torch_chords7_dataset import Chords7Data

from hf_tempi_dataset import STHFTempiData
from hf_chords_dataset import STHFChordsData
from hf_timesig_dataset import STHFTimeSignaturesData
from hf_simpleprog_dataset import STHFSimpleProgressionsData

device = 'cpu'

datadict  = UD.load_data_dict('modemix_chordprog', classify_by_subcategory = False, use_folds = True)

cur_ds = ModemixChordprogData(datadict['df'], embedding_type = 'mg_audio', device=device, layer_idx=0, classify_by_subcategory = False, is_64bit = False, is_memmap = True)


label_arr = datadict['label_arr']
cur_subset = UP.torch_get_train_test_subsets(cur_ds, datadict['df'] , label_arr, train_on_middle = False, train_pct = 0.7, test_subpct = 0.5,seed = 5, debug=False, debug_name='', use_folds = True)
tr_ds = cur_subset['train']
v_ds = cur_subset['valid']
te_ds = cur_subset['test']



def _objective(dataset = 'modemix_chordprog', embedding_type = 'mg_audio', is_classification = True, thresh=0.01, layer_idx = -1, train_ds = None, valid_ds = None,  train_on_middle = False, classify_by_subcategory = False, model_type='musicgen-large', model_layer_dim=128, out_dim = 1, prune=False, num_layers = 1, num_epochs=100):
    train_ds.dataset.set_layer_idx(0)
    valid_ds.dataset.set_layer_idx(0)


    scaler = TST(with_mean = True, with_std = True, dim=model_layer_dim, use_64bit = True, use_constant_feature_mask = True, device = device)
    scaler2 = KST()
    batch_size = 64
    held_out_classes = False
    is_close = True
    for epoch_idx in range(num_epochs):
        train_dl = TUD.DataLoader(train_ds, batch_size = batch_size, shuffle=True, generator=torch.Generator(device=device))
        for data_idx, data in enumerate(train_dl):
            _ipt, ground_truth = data
            scaler.partial_fit(_ipt)
            ipt = scaler.transform(_ipt)
            scaler2.partial_fit(_ipt.numpy())
            ipt2 = scaler2.transform(_ipt)
            is_close = np.allclose(ipt.numpy(), ipt2)
            if is_close == False:
                print('not is close')
                break
        if is_close == False:
            break
    return is_close    
_is_close = _objective(train_ds = tr_ds, valid_ds = v_ds)
if _is_close == False:
    print('not is close')
else:
    print('is close!')


