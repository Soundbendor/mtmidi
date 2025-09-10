import torch
import tomllib
import util as UM
import polars as pl
import os
import numpy as np
from sklearn.model_selection import train_test_split
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


THRESH = 0.1

def get_save_ext(is_memmap):
    save_ext = 'npy'
    if is_memmap == True:
        save_ext = 'dat'
    return save_ext


def pl_init(cur_df, is_classification):
    _ = PL.init(cur_df,is_classification) 


def flatten_toml_dict(toml_dict):
    ret = {}
    for big_k,k_dict in toml_dict.items():
        for k,v in k_dict.items():
            rec_str = f'{big_k}_{k}'
            if type(v) == type([]):
                ret[rec_str] = ",".join([str(cur) for cur in v])
            else:
                ret[rec_str] = v
    return ret

# train_pct refers to entire dataset, test_subpct refers to length after split
def get_train_test_subsets(dataset_label_arr, train_on_middle = True, train_pct = 0.7, test_subpct = 0.5, seed = 5):
    test_valid_pct = 1. - train_pct
    valid_pct = (1. - test_subpct) * test_valid_pct
    test_pct = test_subpct * test_valid_pct
    train_idx = []
    test_valid_idx = []
    total_num = len(dataset_label_arr)
    all_idx = np.arange(0, total_num)
    if train_on_middle == False:
        _train_idx = []
        _test_valid_idx = []
        print('not training on middle')
        if train_pct < 1.0:
            print(f'splitting train/test with train_size = {train_pct}')
            _train_idx, _test_valid_idx = train_test_split(all_idx, random_state = seed, train_size = train_pct, shuffle = True, stratify=dataset_label_arr)
        train_idx = np.array(_train_idx, dtype=int)
        test_valid_idx = np.array(_test_valid_idx, dtype=int)

    else:
        # getting start index of train, starting with valid_pct arbitrarily
        train_start = int(valid_pct * total_num)
        train_end = int((1. - test_pct) * total_num)
        train_idx =  all_idx[train_start:train_end]
        test_valid_idx = np.concatenate((all_idx[:train_start],all_idx[train_end:]))
    leftover_labels = dataset_label_arr[test_valid_idx]
    # returns indices of our index lists so we have to convert to regular indices
    if train_pct < 1.0:
        if test_subpct < 1.0:
            print(f'splitting test/valid with test_size = {test_subpct}')
            test_idx, valid_idx = train_test_split(test_valid_idx, train_size = test_subpct, random_state = seed, shuffle= True, stratify=leftover_labels)
        else:
            test_idx = test_valid_idx
    #test_idx = test_valid_idx[_test_idx]
    #valid_idx = test_valid_idx[_valid_idx]
    idxs = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    return idxs



def read_toml_file(cur_fname):
    data = None
    with open(os.path.join(UM.by_projpath('toml'), f'{cur_fname}.toml'), 'rb') as f:
        data = tomllib.load(f)
    return data

def get_exclude_col_vals(toml_dict):
    ret = []
    if 'exclude' in toml_dict.keys():
        for x_key, x_arr in toml_dict['exclude'].items():
            cur_col = (x_key, x_arr)
            ret.append(cur_col)
    return ret

def exclude_col_vals_in_data(dframe, exclude):
    if len(exclude) < 1:
        return dframe
    else:
        for cur_col, exclude_arr in exclude:
            all_col_vals = dframe.select([cur_col]).to_numpy().flatten()
            dframe = dframe.filter((pl.col(cur_col).is_in(np.setdiff1d(all_col_vals, exclude_arr))))
        return dframe


def get_toml_params(toml_dict):
    ret = {}
    if 'params' in toml_dict.keys():
        ret = toml_dict['params']
    return ret


def get_df(dataset, exclude_arr):
    csvpath = None
    if dataset in UM.new_datasets:
        csvpath = os.path.join(UM.by_projpath('csv', make_dir = False), f'{dataset}.csv')
    elif dataset in UM.hf_datasets:
        csvpath = os.path.join(UM.by_projpath('hf_csv', make_dir = False), f'{dataset}.csv')
    cur_data = pl.read_csv(csvpath)
    cur_data = exclude_col_vals_in_data(cur_data, exclude_arr)
    return cur_data



  


def load_data_dict(cur_dsname, classify_by_subcategory = False, tomlfile_str = ''):
    toml_dict = None
    exclude_arr = None
    num_classes = 1
    thresh = 0.1
    exclude = []
    label_arr = []
    using_toml = False
    if len(tomlfile_str) > 0:
        using_toml = True
        toml_dict = read_toml_file(tomlfile_str)
        exclude = get_exclude_col_vals(toml_dict)
        toml_params = get_toml_params(toml_dict)
        thresh = toml_params.get('thresh', THRESH)

    # get dataframe from csv file
    cur_df = None
    # exclude all intervals besides root position
    if cur_dsname in UM.chordprog_datasets: 
        exclude_arr = []
        if cur_dsname == 'secondary_dominant':
            exclude_arr.append(('inv', [1,2,3]))
        if cur_dsname == 'modemix_chordprog':
            exclude_arr.append(('inv', [1,2]))

        cur_df=get_df(cur_dsname, exclude_arr)
    else:
        # get dataframe from csv file
        cur_df=get_df(cur_dsname, exclude)
   
    is_classification = cur_dsname not in UM.reg_datasets
    pl_classdict = None
    label_col = None
    if cur_dsname == 'polyrhythms':
        # hacky way of initing globals for metrics (but don't overwrite cur_df) now
        pl_init(cur_df, is_classification)
        cur_df = PL.init(cur_df, is_classification)

        if is_classification == True:
            num_classes = PL.num_poly

            pl_classdict = PL.polystr_to_idx
        else:
            pl_classdict = PL.reg_polystr_to_idx
        label_col = 'poly'
    elif cur_dsname == 'tempos':
        label_col = 'bpm_class'

    elif cur_dsname == 'dynamics':
        if classify_by_subcategory == True:
            num_classes = DYN.num_subcategories
            label_col = 'dyn_subcategory'
            pl_classdict = DYN.dyn_subcategory_to_idx
        
        else:
            num_classes = DYN.num_categories
            label_col = 'dyn_category'
            pl_classdict = DYN.dyn_category_to_idx
    
    elif cur_dsname == 'chords7':
        num_classes = CH7.num_chords
        pl_classdict = CH7.quality_to_idx 
        label_col = 'quality'

    elif cur_dsname == 'chords':
        num_classes = HFC.num_chords
        label_col = 'chord_type'

    elif cur_dsname == 'time_signatures':
        num_classes = HTS.num_timesig
        label_col = 'time_signature'
    
    elif cur_dsname == 'simple_progressions':
        if classify_by_subcategory == True:
            num_classes = HFSP.num_progs
            label_col = 'orig_prog'
        else:
            num_classes = HFSP.num_types
            label_col = 'is_major'


    elif cur_dsname == 'modemix_chordprog':
        if classify_by_subcategory == True:
            num_classes = CHP.num_subprog
            pl_classdict = CHP.subp_to_idx
            label_col = 'sub_prog'
        else:
            num_classes = CHP.num_ismodemix
            pl_classdict = CHP.imm_to_idx
            label_col = 'is_modemix'

    elif cur_dsname == 'secondary_dominant':
        if classify_by_subcategory == True:
            num_classes = CSP.num_subprog
            pl_classdict = CSP.subp_to_idx
            label_col = 'sub_prog'
        else:
            num_classes = CSP.num_subtypes
            label_col = 'sub_type'
            pl_classdict = CSP.sub_type_to_idx
    label_arr = cur_df.select([label_col]).to_numpy().flatten()


 
    ret = {}
    ret['num_classes'] = num_classes
    ret['df'] = cur_df
    ret['toml_dict'] = toml_dict
    ret['pl_classdict'] = pl_classdict
    ret['is_classification'] = is_classification
    ret['thresh'] = thresh
    ret['label_col'] = label_col
    ret['using_toml'] = using_toml
    ret['label_arr'] = label_arr
    return ret


def collate_data_at_idx(cur_df,layer_idx, emb_type, is_memmap = True, acts_folder = 'acts', dataset = 'polyrhythms', to_torch = False, use_64bit = False, use_shape = None, device = 'cpu'):
    cur_names = cur_df['name']
    cur_acts = [get_data_vec_at_idx(cur_name, layer_idx, emb_type, is_memmap = is_memmap, acts_folder = acts_folder, dataset = dataset, to_torch = to_torch, use_64bit = use_64bit, use_shape = use_shape, device = device) for cur_name in cur_names]
    if to_torch == False:
        return cur_acts
    else:
        return torch.vstack(cur_acts)


def get_data_vec_at_idx(fname, layer_idx, emb_type, is_memmap = True, acts_folder = 'acts', dataset = 'polyrhythms', to_torch = False, use_64bit = False, use_shape = None, device = 'cpu'):
    cur = None
    if is_memmap == True:
        cur_fname = f'{fname}.dat'
        emb_file = UM.get_embedding_file(emb_type, acts_folder = acts_folder, dataset=dataset, fname=cur_fname, write = False, use_64bit = use_64bit, use_shape = use_shape)
        if layer_idx >= 0:
            cur = emb_file[layer_idx,:].copy()
        else:
            cur = emb_file.copy()
    else: 
        cur_fname = f'{fname}.npy'
        actpath = UM.by_projpath(acts_folder)
        datapath = os.path.join(actpath, dataset)
        modelpath = os.path.join(datapath, emb_type)
        fpath = os.path.join(modelpath, cur_fname)
        #arr = np.load(fpath, allow_pickle == True)
        arr = np.load(fpath)
        #print(arr.shape, layer_idx)
        cur = None
        if layer_idx >= 0 and len(arr.shape) > 1:
            cur = arr[layer_idx,:]
        else:
            cur = arr
        if cur.dtype == np.float32:
            if use_64bit == True:
                cur = cur.astype(np.float64)
        elif cur.dtype == np.float64:
            if use_64bit == False:
                cur = cur.astype(np.float32)
    if to_torch == True:
        cur = torch.from_numpy(cur).to(device)
    return cur

