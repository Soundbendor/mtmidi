import sklearn.metrics as SKM
import torch
import torch.utils.data as TUD
import util_data as UD
import numpy as np
import polyrhythms as PL
import dynamics as DYN
import chords7 as CH7
import chordprog as CHP
import chord7prog as CSP
import hf_chords as HFC
import hf_timesig as HTS
import hf_simpleprog as HFSP
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

scaler_dir = UM.by_projpath(subpath = 'saved_scalers', make_dir = True)

def init(class_binsize):
    TP.init(class_binsize)

def record_dict_in_study(study, flat_dict):
    for k,v in flat_dict.items():
        study.set_user_attr(k,v)

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
        elif dataset == 'modemix_chordprog':
            if classify_by_subcategory == True:
                class_truths = [CHP.idx_to_subp[x] for x in truths]
                class_preds = [CHP.idx_to_subp[x] for x in preds]
                class_arr = CHP.subp_arr
            else:
                class_truths = [CHP.idx_to_imm[x] for x in truths]
                class_preds = [CHP.idx_to_imm[x] for x in preds]
                class_arr = CHP.is_modemix_arr

        elif dataset == 'secondary_dominant':
            if classify_by_subcategory == True:
                class_truths = [CSP.idx_to_subp[x] for x in truths]
                class_preds = [CSP.idx_to_subp[x] for x in preds]
                class_arr = CSP.subp_arr
            else:
                class_truths = [CSP.idx_to_sub_type[x] for x in truths]
                class_preds = [CSP.idx_to_sub_type[x] for x in preds]
                class_arr = CSP.sub_type_arr
        
        elif dataset == 'simple_progressions':
            if classify_by_subcategory == True:
                class_truths = [HFSP.idx_to_prog[x] for x in truths]
                class_preds = [HFSP.idx_to_prog[x] for x in preds]
                class_arr = HFSP.prog_arr
            else:
                class_truths = [HFSP.idx_to_major[x] for x in truths]
                class_preds = [HFSP.idx_to_major[x] for x in preds]
                class_arr = HFSP.major_arr


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

def torch_get_train_test_subsets(dataset_obj, dataset_df, dataset_label_arr, train_on_middle = True, train_pct = 0.7, test_subpct = 0.5, seed = 5, debug=False, debug_name='', use_folds = True):
    idxs = None
    if use_folds == False:
        idxs = UD.get_train_test_subsets(dataset_label_arr, train_on_middle = train_on_middle, train_pct = train_pct, test_subpct = test_subpct, seed = seed)
    else:
        num_ex = len(dataset_df)
        idx_arr = np.arange(num_ex, dtype=int)
        cur_df = dataset_df.with_columns(**{'ex_idx': idx_arr})
        idxs = {}
        for set_type in ['train', 'valid', 'test']:
            cur_idxs = cur_df.filter(pl.col('set_type') == set_type)['ex_idx']
            idxs[set_type] = cur_idxs.to_numpy()


    if debug == True:
        dest_dir = UM.by_projpath(subpath='split_debug', make_dir = True)
        for idxtype in ['train', 'valid', 'test']:
            out_file = f'{debug_name}-{idxtype}.txt'
            dest_file = os.path.join(dest_dir, outfile)
            with open(dest_file, 'w') as f:
                out_str = ' '.join([str(_idx) for _idx in idxs[idxtype]])
                f.write(out_str)


    valid_subset = None
    test_subset = None
    train_subset = TUD.Subset(dataset_obj, idxs['train'])
    if len(idxs['valid']) > 0:
        valid_subset = TUD.Subset(dataset_obj, idxs['valid'])
    if len(idxs['test']) > 0:
        test_subset = TUD.Subset(dataset_obj, idxs['test'])
    subsets = {'train': train_subset, 'test': test_subset, 'valid': valid_subset}
    return subsets



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

def save_scaler(scaler, model_shorthand = 'mg_large_h', dataset = 'polyrhythms', prefix=5, trial_number = 1, is_64bit = True):
    save_dir = UM.get_model_save_path(model_shorthand, dataset=dataset, return_relative = False, make_dir = True)
    cur_ext = None
    if is_64bit == True:
        cur_ext = '64.scaler_dict'
    else:
        cur_ext = '32.scaler_dict'
    out_path = os.path.join(save_dir, f'{prefix}-{trial_number}-{cur_ext}')
    torch.save(scaler.state_dict(), out_path)

def load_scaler(scaler, model_shorthand = 'mg_large_h', dataset = 'polyrhythms', prefix=5, trial_number = 1, is_64bit = True):
    save_dir = UM.get_model_save_path(model_shorthand, dataset=dataset, return_relative = False, make_dir = True)
    cur_ext = None
    if is_64bit == True:
        cur_ext = '64.scaler_dict'
    else:
        cur_ext = '32.scaler_dict'
    out_path = os.path.join(save_dir, f'{prefix}-{trial_number}-{cur_ext}')

    scaler.load_state_dict(torch.load(out_path, weights_only = True))

def save_probe(model, model_shorthand = 'mg_large_h', dataset = 'polyrhythms', prefix=5, trial_number = 1):
    save_dir = UM.get_model_save_path(model_shorthand, dataset=dataset, return_relative = False, make_dir = True)
    out_path = os.path.join(save_dir, f'{prefix}-{trial_number}.probe_dict')

    torch.save(model.state_dict(), out_path)

def load_probe(model, model_shorthand = 'mg_large_h', dataset = 'polyrhythms', prefix=5, trial_number = 1):
    save_dir = UM.get_model_save_path(model_shorthand, dataset=dataset, return_relative = False, make_dir = True)
    out_path = os.path.join(save_dir, f'{prefix}-{trial_number}.probe_dict')

    model.load_state_dict(torch.load(out_path, weights_only = True))
