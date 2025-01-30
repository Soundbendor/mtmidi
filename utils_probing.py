import sklearn.metrics as SKM
import torch.utils.data as TUD
from sklearn.model_selection import train_test_split
import numpy as np
import polyrhythms as PL
import dynamics as DYN
import matplotlib.pyplot as plt
import util as UM
import neptune
import tempi as TP
import os
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

def get_classification_metrics(truths, preds, dataset = 'polyrhythms', classify_by_subcategory = False, save_confmat=True):
    acc = SKM.accuracy_score(truths, preds)
    f1_macro = SKM.f1_score(truths, preds, average='macro')
    f1_micro = SKM.f1_score(truths, preds, average='micro')
    class_truths = None
    class_preds = None
    if dataset == 'polyrhythms':
        class_truths = [PL.rev_polystr_to_idx[x] for x in truths]
        class_preds = [PL.rev_polystr_to_idx[x] for x in preds]
    elif dataset == 'dynamics':
        if classify_by_subcategory == True:
            class_truths = [DYN.dyn_idx_to_subcategory[x] for x in truths]
            class_preds = [DYN.dyn_idx_to_subcategory[x] for x in preds]
        else:
            class_truths = [DYN.dyn_idx_to_category[x] for x in truths]
            class_preds = [DYN.dyn_idx_to_category[x] for x in preds]
    cm = None
    cm_path = None
    if save_confat == True:
        cm = SKM.confusion_matrix(class_truths, class_preds)
        cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=PL.class_arr)
        fig, ax = plt.subplots(figsize=(figsize,figsize))
        cmd.plot(ax=ax)
        timestamp = int(time.time()*1000)

        cm_fname = f'{timestamp}-cm.png' 
        cm_path = os.path.join(res_dir, cm_fname)
        plt.savefig(cm_path)
        plt.clf()
    d = {'accuracy_score': acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'confmat': cm, 'confmat_path': cm_path}
    return d

def get_regression_metrics(truths, truth_labels, preds, pred_labels, dataset = 'polyrhythms',  held_out_classes = False, save_confmat = True, do_regression_classification = True):
    mse = SKM.mean_squared_error(truths, preds)
    r2 = SKM.r2_score(truths, preds)
    mae = SKM.mean_absolute_error(truths, preds)
    ev = SKM.explained_variance_score(truths, preds)
    medae = SKM.median_absolute_error(truths, preds)
    maxerr = SKM.max_error(truths, preds)
    mape = SKM.mean_absolute_percentage_error(truths, preds)
    rmse = SKM.root_mean_squared_error(truths, preds)
    d2ae = SKM.d2_absolute_error_score(truths, preds)
    acc = None
    f1_macro = None
    f1_micro = None
    if do_regression_classification == True:
        acc = SKM.accuracy_score(truth_labels, pred_labels)
        f1_macro = SKM.f1_score(truth_labels, pred_labels, average='macro')
        f1_micro = SKM.f1_score(truth_labels, pred_labels, average='micro')

    
    class_truths = None
    class_preds = None
    cm = None
    cm_path = None
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
        train_idx = np.array(train_idx)
        test_valid_idx = np.array(test_valid_idx)
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



     
# log test results to neptune
def neptune_log(nep, results_dict):
    for res_key, res_val in results_dict.items():
        if np.any(res_val) != None:
            if res_key not in nep_dont_log:
                if res_key in nep_paths:
                    split_key = res_key.split("_")[0]
                    nep_key = f"test/{split_key}"
                    nep[nep_key].upload(res_val)
                else:
                    nep_key = f"test/{res_key}"
                    nep[nep_key] = res_val

            
