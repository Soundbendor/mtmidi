import sklearn.metrics as SKM
import torch.utils.data as TUD
from sklearn.model_selection import train_test_split
import numpy as np
import polyrhythms as PL
import dynamics as DYN
import chords7 as CH7
import hf_chords as HFC
import hf_timesig as HTS
import matplotlib.pyplot as plt
import util as UM
import neptune
import tempi as TP
import os, csv
import time
import tomllib
import polars as pl

figsize = 15

res_dir = UM.by_projpath("res", make_dir = True)
# do not log these metrics because it won't work with neptune anyways
nep_dont_log = set(['confmat'])
# dont need to log this either, use to upload
nep_paths = set(['confmat_path'])

def init(class_binsize):
    TP.init(class_binsize)

def pl_init(cur_df, is_classification):
    _ = PL.init(cur_df,is_classification) 

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

def record_dict_in_study(study, flat_dict):
    for k,v in flat_dict.items():
        study.set_user_attr(k,v)

def flatten_toml_dict(toml_dict):
    ret = {}
    for big_k,k_dict in toml_dict.items():
        for k,v in k_dict.items():
            rec_str = f'{big_k}_{k}'
            ret[rec_str] = v
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

def get_classification_metrics(truths, preds, dataset = 'polyrhythms', classify_by_subcategory = False, save_confmat=True, file_basename = None):

    acc = SKM.accuracy_score(truths, preds)
    f1_macro = SKM.f1_score(truths, preds, average='macro')
    f1_micro = SKM.f1_score(truths, preds, average='micro')
    


    cm = []
    cm_path = []
    if save_confmat == True:
        class_truths = []
        class_preds = []
        class_arr = None
        if dataset == 'polyrhythms':
            class_truths = [PL.rev_polystr_to_idx[x] for x in truths]
            class_preds = [PL.rev_polystr_to_idx[x] for x in preds]
            class_arr = PL.class_arr
        elif dataset == 'dynamics':
            if classify_by_subcategory == True:
                class_truths = [DYN.dyn_idx_to_subcategory[x] for x in truths]
                class_preds = [DYN.dyn_idx_to_subcategory[x] for x in preds]
                class_arr = DYN.dyn_subcategories
            else:
                class_truths = [DYN.dyn_idx_to_category[x] for x in truths]
                class_preds = [DYN.dyn_idx_to_category[x] for x in preds]
                class_arr = DYN.dyn_categories
        elif dataset == 'chords7':
            class_truths = [CH7.idx_to_quality[x] for x in truths]
            class_preds = [CH7.idx_to_quality[x] for x in preds]
            class_arr = CH7.class_arr
        elif dataset == 'chords':
            class_truths = [HFC.idx_to_quality[x] for x in truths]
            class_preds = [HFC.idx_to_quality[x] for x in preds]
            class_arr = HFC.class_arr
        elif dataset == 'time_signatures':
            class_truths = [HTS.idx_to_timesig[x] for x in truths]
            class_preds = [HTS.idx_to_timesig[x] for x in preds]
            class_arr = HTS.class_arr

        cm = SKM.confusion_matrix(class_truths, class_preds)
        cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_arr)
        fig, ax = plt.subplots(figsize=(figsize,figsize))
        cmd.plot(ax=ax)
        timestamp = int(time.time()*1000)
        cm_fname = f'{timestamp}-cm.png' 
        if file_basename != None:
            cm_fname = f'{file_basename}-cm.png'
        cm_path = os.path.join(res_dir, cm_fname)
        plt.savefig(cm_path)
        plt.clf()
    d = {'accuracy_score': acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'confmat': cm, 'confmat_path': cm_path}
    return d

def get_regression_metrics(truths, truth_labels, preds, pred_labels, dataset = 'polyrhythms',  held_out_classes = False, save_confmat = True, do_regression_classification = True, file_basename = None):
    mse = SKM.mean_squared_error(truths, preds)
    r2 = SKM.r2_score(truths, preds)
    mae = SKM.mean_absolute_error(truths, preds)
    ev = SKM.explained_variance_score(truths, preds)
    medae = SKM.median_absolute_error(truths, preds)
    maxerr = SKM.max_error(truths, preds)
    mape = SKM.mean_absolute_percentage_error(truths, preds)
    rmse = SKM.root_mean_squared_error(truths, preds)
    d2ae = SKM.d2_absolute_error_score(truths, preds)
    acc = []
    f1_macro = []
    f1_micro = []
    if do_regression_classification == True:
        acc = SKM.accuracy_score(truth_labels, pred_labels)
        f1_macro = SKM.f1_score(truth_labels, pred_labels, average='macro')
        f1_micro = SKM.f1_score(truth_labels, pred_labels, average='micro')

    
    class_truths = [] 
    class_preds = []
    cm = []
    cm_path = []
    if do_regression_classification == True and save_confmat == True:
        if dataset == 'polyrhythms':
            class_truths = [PL.reg_rev_polystr_to_idx[x] for x in truth_labels]
            class_preds = [PL.reg_rev_polystr_to_idx[x] for x in pred_labels]
        elif dataset == 'tempi':
            class_truths = [TP.rev_classdict[x] for x in truth_labels]
            class_preds = [TP.rev_classdict[x] for x in pred_labels]
        cm = SKM.confusion_matrix(class_truths, class_preds)
        cmd = None
        if held_out_classes == True or dataset != 'polyrhythms':
            all_labels = set(class_truths).union(set(class_preds))
            class_arr2 = None
            if dataset == 'polyrhythms':
                class_arr2 = [x for x in PL.reg_class_arr if x in all_labels]
            elif dataset == 'tempos':
                class_arr2 = [x for x in TP.classset_aug if x in all_labels]
            cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_arr2)
        else:
            cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=PL.reg_class_arr)
        cmd.plot()
        fig, ax = plt.subplots(figsize=(figsize,figsize))
        cmd.plot(ax=ax)
        timestamp = int(time.time()*1000)
        cm_fname = f'{timestamp}-cm.png' 
        if file_basename != None:
            cm_fname = f'{file_basename}-cm.png'
        cm_path = os.path.join(res_dir, cm_fname)
        plt.savefig(cm_path)
        plt.clf()

    d = {'mean_squared_error': mse, "r2_score": r2, "mean_absolute_error": mae,
         "explained_variance_score": ev, "median_absolute_error": medae,
         "max_error": maxerr, "mean_absolute_percentage_error": mape,
         "root_mean_squared_error": rmse, "d2_absolute_error_score": d2ae,
         "accuracy_score": acc, "f1_macro": f1_macro, "f1_micro": f1_micro,
         "confmat": cm, 'confmat_path': cm_path}
    return d




# train_pct refers to entire dataset, test_subpct refers to length after split
def get_train_valid_test_subsets(dataset_obj, dataset_label_arr, train_on_middle = True, train_pct = 0.7, test_subpct = 0.5, seed = 5):
    test_valid_pct = 1. - train_pct
    valid_pct = (1. - test_subpct) * test_valid_pct
    test_pct = test_subpct * test_valid_pct
    train_idx = None
    test_valid_idx = None
    total_num = len(dataset_obj)
    all_idx = np.arange(0, total_num)
    if train_on_middle == False:
        _train_idx, _test_valid_idx = train_test_split(all_idx, random_state = seed, shuffle = True, stratify=dataset_label_arr)
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
    test_idx, valid_idx = train_test_split(test_valid_idx, random_state = seed, shuffle= True, stratify=leftover_labels)
    #test_idx = test_valid_idx[_test_idx]
    #valid_idx = test_valid_idx[_valid_idx]
    train_subset = TUD.Subset(dataset_obj, train_idx)
    valid_subset = TUD.Subset(dataset_obj, valid_idx)
    test_subset = TUD.Subset(dataset_obj, test_idx)
    return train_subset, valid_subset, test_subset

def print_metrics(results_dict, study_name, filehandle = None):
    if filehandle != None:
        print(study_name)
    else:
        print(study_name, file=filehandle)
    for res_key, res_val in results_dict.items():
        if res_key not in nep_dont_log and res_key not in nep_paths:
            print_str = f"{res_key}: {res_val}"
            if filehandle != None:
                print(print_str)
            else:
                print(print_str, file=filehandle)


def save_results_to_study(study, results_dict):
    for res_key, res_val in results_dict.items():
        if res_key not in nep_dont_log:
            study_key = f'test_{res_key}'
            study.set_user_attr(study_key, res_val)
                
# filter out none or 0 len results from dict
# if using replace_val, replace instead of filter out
def filter_dict(results_dict, replace_val = None, filter_nonstr = False, keys_dont_log = nep_dont_log):
    ret = {}
    for res_key, res_val in results_dict.items():
        to_log = True
        if res_key not in keys_dont_log:
            if hasattr(res_val, '__len__'):
                if len(res_val) <= 0:
                    to_log = False
                elif filter_nonstr == True:
                    if type(res_val) != type(""):
                        to_log = False
            else:
                if not res_val:
                    to_log = False
        else:
            to_log = False
        if to_log == True:
            ret[res_key] = res_val
        elif replace_val:
            ret[res_key] = replace_val
    return ret


# log test results to neptune
def neptune_log(nep, filt_dict):
    for res_key, res_val in filt_dict.items():
        if res_key in nep_paths:
            split_key = res_key.split("_")[0]
            nep_key = f"test/{split_key}"
            nep[nep_key].upload(res_val)
        else:
            nep_key = f"test/{res_key}"
            nep[nep_key] = res_val


# to work on: log to csv
def log_results(filt_dict, study_name, folder='res_csv'):
    cur_folder = UM.by_projpath(folder,make_dir = True)
    out_path = os.path.join(cur_folder, f'{study_name}.csv')
    cur_header = list(filt_dict.keys())
    f = open(out_path, 'w')
    csvw = csv.DictWriter(f, fieldnames=cur_header)
    csvw.writeheader()
    csvw.writerow(filt_dict)
    f.close()
    
