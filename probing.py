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
import hf_notes as HFN
import hf_scales as HSCL
import hf_intervals as HFI
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

from torch_probe_model import LinearProbe
# global declarations (hacky) to save model state dicts
global trial_model_state_dict
global best_model_state_dict
global study_sampler_path

### init stuff
train_pct = 0.7
test_subpct = 0.5
seed = 5
torch.manual_seed(seed)
shuffle = True
# neptune stuff
plots_update_freq = 10
log_plot_slice = False
log_plot_contour = False
# hacky way of initialize tempo things with a class_binsize for "classification" from regression
TEMPOS_CLASS_BINSIZE=4

TP.init(TEMPOS_CLASS_BINSIZE)
UP.init(TEMPOS_CLASS_BINSIZE)

# cuda stuff
device ='cpu'
if torch.cuda.is_available() == True:
    device = 'cuda'
    torch.cuda.empty_cache()
    torch.set_default_device(device)

### REGRESSION "CLASSIFICATION"
def regression_classification(dataset, predictions, thresh=0.01):
    # predictions should be in numpy format
    cur_pred_labels = None
    cur_pred_label_idx = None
    if dataset == 'polyrhythms':
        cur_pred_labels = [PL.get_nearest_poly(x, thresh=thresh) for x in predictions]
        cur_pred_label_idx = np.array([PL.reg_polystr_to_idx[x] for x in cur_pred_labels])
    elif dataset == 'tempos':
        # this maps normed predictions to bpm classes (middles of bpm bins)
        cur_pred_labels = [TP.get_nearest_bpmclass(x, TP.classlist_sorted, thresh=thresh) for x in predictions]
        #print('test2', pred_np.shape, pred_np.dtype, len(cur_pred_labels))
        # this maps middles of bpm bins to indices 
        cur_pred_label_idx = np.array([TP.classdict[x] for x in cur_pred_labels])
    return cur_pred_labels, cur_pred_label_idx

def train_loop(model, opt_fn, loss_fn, train_ds, batch_size = 64, shuffle = True, is_classification = True, scaler = None):

    train_dl = TUD.DataLoader(train_ds, batch_size = batch_size, shuffle=shuffle, generator=torch.Generator(device=device))
    model.train(True)
    iters = 0
    total_loss = 0
    for data_idx, data in enumerate(train_dl):
        loss = None
        if is_classification == True:
            _ipt, ground_truth = data
            ipt = None
            if scaler != None:
                scaler.partial_fit(_ipt)
                ipt = scaler.transform(_ipt)
            else:
                ipt = _ipt
            pred = model(ipt)
            loss = loss_fn(pred, ground_truth)
        else:
            _ipt, ground_truth, ground_label = data
            ipt = None
            if scaler != None:
                scaler.partial_fit(_ipt)
                ipt = scaler.transform(_ipt)
            else:
                ipt = _ipt
            pred = model(ipt)
            loss = loss_fn(pred.flatten(), ground_truth.flatten())
     
        loss.backward()
        opt_fn.step()
        cur_loss = loss.item()
        total_loss += cur_loss
        iters += 1
    avg_loss = total_loss/float(iters)
    return avg_loss

def valid_test_loop(model, eval_ds, loss_fn = None, dataset = 'polyrhythms', is_classification = True, held_out_classes = False, is_testing = False, batch_size = 64, shuffle = True,thresh = 0.01, classify_by_subcategory = False, file_basename=None, scaler = None):
    eval_dl = TUD.DataLoader(eval_ds, batch_size = batch_size, shuffle=shuffle, generator=torch.Generator(device=device))
    model.eval()
    iters = 0
    total_loss = 0

    # accumulate ground truths and predictions
    truths = None
    preds = None
    # accumulate regression "classification" ground truths and predictions
    truth_labels = None
    pred_labels = None
    for data_idx, data in enumerate(eval_dl):
        loss = None
        if is_classification == True:
            _ipt, ground_truth = data
            ipt = None
            if scaler != None:
                ipt = scaler.transform(_ipt)
            else:
                ipt = _ipt

            pred = model(ipt)
            if loss_fn != None:
                loss = loss_fn(pred, ground_truth)
            
            cur_truths = ground_truth.detach().cpu().numpy().flatten()
            cur_preds = torch.argmax(pred,axis=1).detach().cpu().numpy().flatten()
            
            if data_idx == 0:
                #truths = copy.deepcopy(cur_truths)
                #preds = copy.deepcopy(cur_preds)
                truths = cur_truths
                preds = cur_preds
            else:
                if truths.base is None:
                    truths = np.hstack((truths, cur_truths))
                else:
                    truths = np.hstack((truths, copy.deepcopy(cur_truths)))
                if preds.base is None: 
                    preds = np.hstack((preds, cur_preds))
                else:
                    preds = np.hstack((preds, copy.deepcopy(cur_preds)))
        else:
            _ipt, ground_truth, ground_label = data
            ipt = None
            if scaler != None:
                ipt = scaler.transform(_ipt)
            else:
                ipt = _ipt
            pred = model(ipt)
            if loss_fn != None:
                loss = loss_fn(pred.flatten(), ground_truth.flatten())
            # stuff for regression "classification"  
            pred_np = pred.detach().cpu().numpy().flatten()
            #if do_regression_classification == True:
            #cur_pred_labels, cur_pred_label_idx = regression_classification(dataset, pred_np, thresh=thresh)
            if data_idx == 0:
                #preds = copy.deepcopy(copy.deepcopy(pred_np))
                preds = pred_np
                #truths = copy.deepcopy(ground_truth.detach().cpu().numpy().flatten())
                truths = ground_truth.detach.cpu().numpy().flatten()
                #if do_regression_classification == True:
                #truth_labels = copy.deepcopy(ground_label.detach().cpu().numpy().flatten())
                #pred_labels = copy.deepcopy(cur_pred_label_idx)
            else:
                if preds.base is None:
                    preds = np.hstack((preds, pred_np))
                else:
                    preds = np.hstack((preds, copy.deepcopy(pred_np)))
                if truths.base is None:
                    truths = np.hstack((truths, ground_truth.detach().cpu().numpy().flatten()))
                else:
                    truths = np.hstack((truths, copy.deepcopy(ground_truth.detach().cpu().numpy().flatten())))
                #if do_regression_classification == True:
                #truth_labels = np.hstack((truth_labels, copy.deepcopy(ground_label.detach().cpu().numpy().flatten())))
                #pred_labels = np.hstack((pred_labels, copy.deepcopy(cur_pred_label_idx)))

            #print('truth', data_idx, truth_labels.shape)
            #print('pred', data_idx, pred_labels.shape)
        # loss bookkeeping
        if loss_fn != None:
            cur_loss = loss.item()
            total_loss += cur_loss
            iters += 1
    # metrics calculating
    metrics = None
    if is_classification == True:
        # only save confusion matrix if testing
        metrics = UP.get_classification_metrics(truths, preds, dataset = dataset, classify_by_subcategory = classify_by_subcategory, save_confmat=is_testing, file_basename=file_basename)
    else:
        metrics = UP.get_regression_metrics(truths, truth_labels, preds, pred_labels, dataset = dataset,  held_out_classes = held_out_classes, save_confmat = is_testing)
    avg_loss = 0
    if loss_fn != None:
        avg_loss = total_loss/float(iters)
    return avg_loss, metrics

# for use with optuna trials
def get_optimization_metric(metric_dict, is_classification = True):
    ret = None
    if is_classification == True:
        ret = metric_dict['accuracy_score']
    else:
        ret = metric_dict['r2_score']
    return ret

def has_held_out_classes(dataset, is_classification):
    return (dataset in UM.tom_datasets) and is_classification == False

def _objective(trial, dataset = 'polyrhythms', embedding_type = 'mg_small_h', is_classification = True, thresh=0.01, layer_idx = -1, train_ds = None, valid_ds = None,  train_on_middle = False, classify_by_subcategory = False, model_type='musicgen-small', model_layer_dim=1024, out_dim = 1, prune=False, num_layers = 1, num_epochs=100, prefix = 1, early_stopping_check_interval = 1, early_stopping_boredom = 5):

    model = None
    # suggested params
    lr_exp = trial.suggest_int('learning_rate_exp',-5,-3, step=1)
    lr = 10**lr_exp
    
    dropout = trial.suggest_float('dropout', 0.25, 0.75, step=0.25)
        
    weight_decay_exp = trial.suggest_int('l2_weight_decay_exp', -4,-2,step=1)
    weight_decay = 10**weight_decay_exp

    batch_size = trial.suggest_categorical('batch_size', [64,256])
    lidx = None
    if layer_idx >= 0:
        lidx = layer_idx
    else:
        lidx_list = list(range(num_layers))
        lidx = trial.suggest_categorical('layer_idx', lidx_list)
        #lidx = trial.suggest_int('layer_idx', 0, num_layers - 1, step=1)

    train_ds.dataset.set_layer_idx(lidx)
    valid_ds.dataset.set_layer_idx(lidx)


    data_norm = trial.suggest_categorical('data_norm', [False, True])
    trial_number = trial.number
    scaler = None
    if data_norm == True:
        scaler = TST(with_mean = True, with_std = True, dim=model_layer_dim, use_64bit = True, use_constant_feature_mask = True, device = device)

    held_out_classes = has_held_out_classes(dataset, is_classification)     
    model = LinearProbe(in_dim=model_layer_dim, hidden_layers = [512],out_dim=out_dim, dropout = dropout, initial_dropout = True)

    # optimizer and loss init
    opt_fn = None
    # count weight decay 10^-2 and bigger as off
    if weight_decay_exp < -2:
        opt_fn = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt_fn = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = None
    if is_classification == True:
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_fn = nn.MSELoss(reduction='mean')

    # polyrhythm and tempi regression has held out classes for "classification"
    #held_out_classes = (dataset in ["polyrhythms", "tempos"]) and is_classification == False


    doing_early_stopping = early_stopping_check_interval > 0

    ret_score = None
    best_score = float('-inf') 

    cur_boredom = 0 

    best_probe_dict= None
    best_scaler_dict = None
    actual_epochs = num_epochs

    for epoch_idx in range(num_epochs):
        train_loss = train_loop(model, opt_fn, loss_fn, train_ds, batch_size = batch_size, is_classification = is_classification, scaler = scaler)
        valid_loss, valid_metrics = valid_test_loop(model,valid_ds, loss_fn = loss_fn, dataset = dataset, is_classification = is_classification, held_out_classes = held_out_classes, is_testing = False, thresh = thresh, batch_size = batch_size, classify_by_subcategory = classify_by_subcategory, scaler = scaler)
        cur_score = get_optimization_metric(valid_metrics, is_classification = is_classification)
        # https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html

        if prune == True: 
            if trial.should_prune() == True:
            
                trial.report(cur_score, epoch_idx)
                raise optuna.TrialPruned()

        if doing_early_stopping == True:
            if epoch_idx % early_stopping_check_interval == 0:
                if cur_score > best_score:
                    best_score = cur_score
                    cur_boredom = 0
                    best_probe_dict = copy.deepcopy(model.state_dict())
                    if scaler != None:
                        best_scaler_dict = copy.deepcopy(scaler.state_dict())
                else:
                    cur_boredom += 1
        if cur_boredom >= early_stopping_boredom:
            actual_epochs = epoch_idx + 1
            ret_score = best_score
            break
        elif epoch_idx == (num_epochs - 1):
            best_probe_dict = copy.deepcopy(model.state_dict())
            ret_score = cur_score
            if scaler != None:
                best_scaler_dict = copy.deepcopy(scaler.state_dict())




    trial.set_user_attr(key='actual_epochs', value=actual_epochs)
    if best_probe_dict != None:
        UP.save_probe_dict(best_probe_dict, model_shorthand = embedding_type, dataset = dataset, prefix=prefix, trial_number = trial_number)
    if best_scaler_dict != None:
        UP.save_scaler_dict(best_scaler_dict, model_shorthand = embedding_type, dataset = dataset, prefix=prefix, trial_number = trial_number)

    return ret_score
    
def study_callback(study, trial):
    global study_sampler_path
    with open(study_sampler_path, 'wb') as f:
        pickle.dump(study.sampler, f)

def eval_train(model, scaler = None, dataset = 'polyrhythms', embedding_type = 'mg_small_h', lr_exp = -3, weight_decay_exp = -2, batch_size = 64, is_classification = True, thresh=0.01, layer_idx = -1, train_ds = None, valid_ds = None,  train_on_middle = False, classify_by_subcategory = False, model_type='musicgen-small', model_layer_dim=1024, out_dim = 1, num_epochs=100, early_stopping_check_interval = 1, early_stopping_boredom = 5):

    lr = 10**lr_exp
    weight_decay = 10**weight_decay_exp

    lidx = layer_idx # hacky way of keeping old code from _objective
    train_ds.dataset.set_layer_idx(lidx)
    valid_ds.dataset.set_layer_idx(lidx)


    held_out_classes = has_held_out_classes(dataset, is_classification)     

    # optimizer and loss init
    opt_fn = None
    # count weight decay 10^-2 and bigger as off
    if weight_decay_exp < -2:
        opt_fn = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt_fn = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = None
    if is_classification == True:
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_fn = nn.MSELoss(reduction='mean')

    # polyrhythm and tempi regression has held out classes for "classification"
    #held_out_classes = (dataset in ["polyrhythms", "tempos"]) and is_classification == False
    doing_early_stopping = early_stopping_check_interval > 0

    ret_score = None
    best_score = float('-inf') 

    cur_boredom = 0 

    actual_epochs = num_epochs

    for epoch_idx in range(num_epochs):
        train_loss = train_loop(model, opt_fn, loss_fn, train_ds, batch_size = batch_size, is_classification = is_classification, scaler = scaler)
        valid_loss, valid_metrics = valid_test_loop(model,valid_ds, loss_fn = loss_fn, dataset = dataset, is_classification = is_classification, held_out_classes = held_out_classes, is_testing = False, thresh = thresh, batch_size = batch_size, classify_by_subcategory = classify_by_subcategory, scaler = scaler)
        cur_score = get_optimization_metric(valid_metrics, is_classification = is_classification)
        # https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html

        if doing_early_stopping == True:
            if epoch_idx % early_stopping_check_interval == 0:
                if cur_score > best_score:
                    best_score = cur_score
                    cur_boredom = 0
                else:
                    cur_boredom += 1
        if cur_boredom >= early_stopping_boredom:
            actual_epochs = epoch_idx + 1
            ret_score = best_score
            break
        elif epoch_idx == (num_epochs - 1):
            ret_score = cur_score
    return ret_score
 


if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-et", "--embedding_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-nt", "--num_trials", type=int, default=-1, help="number of optuna trials")
    parser.add_argument("-li", "--layer_idx", type=int, default=-1, help="< 0 to optimize by optuna, else specifies layer_idx 0-indexed")
    parser.add_argument("-cls", "--is_classification", type=strtobool, default=True, help="is classification")
    parser.add_argument("-tom", "--train_on_middle", type=strtobool, default=False, help="train on middle")
    parser.add_argument("-rc", "--do_regression_classification", type=strtobool, default=False, help="do regression classification")
    parser.add_argument("-nep", "--to_nep", type=strtobool, default=True, help="log on neptune")
    parser.add_argument("-cbs", "--classify_by_subcategory", type=strtobool, default=False, help="classify by subcategory for dynamics, by progression for chord progression datasets")
    parser.add_argument("-pf", "--prefix", type=int, default=-1, help="specify a prefix > 0 for save files (db, etc.) for potential reloading (if file exists)")
    parser.add_argument("-tf", "--toml_file", type=str, default="", help="toml file in toml directory with exclude category listing vals to exclude by col, amongst other settings")
    parser.add_argument("-ev", "--eval", type=strtobool, default=False, help="evalute on best performing params recorded")
    parser.add_argument("-rt", "--eval_retrain", type=strtobool, default=False, help="retrain for eval")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="hacky way of syntax debugging")
    parser.add_argument("-epc", "--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-esb", "--early_stopping_boredom", type=int, default=5, help="epoch intervals for early stopping threshold")
    parser.add_argument("-esci", "--early_stopping_check_interval", type=int, default=1, help="check every epochs for early stopping (<= 0 turns off)")
    parser.add_argument("-spd", "--split_debug", type=strtobool, default=False, help="debug split by recording indices")
    parser.add_argument("-uf", "--use_folds", type=strtobool, default=True, help="use predefined folds for dataset splitting")
    parser.add_argument("-pr", "--prune", type=strtobool, default=False, help="do pruning")
    parser.add_argument("-gr", "--grid_search", type=strtobool, default=True, help="grid search")
    parser.add_argument("-fs", "--full_search", type=strtobool, default=False, help="force full parameter search")
    parser.add_argument("-rs", "--reduced_search", type=strtobool, default=True, help="reduced parameter search")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="load embeddings as memmap, else npy")
    parser.add_argument("-sh", "--on_share", type=strtobool, default=False, help="load from share partition")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")

    # obj_dict is for passing to objective function, is arg_dict without drop_keys
    # rec_dict is for passing to neptune and study (has drop keys)
    # arg_dict just has everything
    drop_keys = set(['to_nep', 'num_trials', 'toml_file', 'do_regression_classification', 'debug', 'memmap', 'slurm_job','grid_search', 'eval', 'split_debug', 'use_folds', 'eval_retrain', 'on_share', 'full_search', 'reduced_search'])
    #### some more logic to define experiments
    args = parser.parse_args()
    arg_dict = vars(args)
    # model type is slightly distinct from embedding_type (which is also shorthand) because musicgen-enocoder uses musicgen-large
    model_type = UM.get_model_type(arg_dict['embedding_type'])  
    model_layer_dim = UM.get_layer_dim(arg_dict['embedding_type'])
   
    # defining grid search
    emb_type = arg_dict['embedding_type']

    #### some variable definitions
    
    is_eval = arg_dict['eval']
    is_64bit = False # if embeddings are 64 bit
    if arg_dict['embedding_type'] in UM.baseline_names:
        is_64bit = False
    cur_ds = None
    label_arr = None
    cur_dsname = arg_dict['dataset']
    train_on_middle = cur_dsname in UM.tom_datasets
    user_specify_layer_idx = arg_dict['layer_idx'] >= 0 
    tomlfile_str = arg_dict['toml_file'] 
    _on_share = arg_dict['on_share'] 
    
    _classify_by_subcategory = arg_dict['classify_by_subcategory'] 
    force_full_search = arg_dict['full_search']
    reduced_search = arg_dict['reduced_search']
    

    datadict  = UD.load_data_dict(cur_dsname, classify_by_subcategory = _classify_by_subcategory, tomlfile_str = tomlfile_str, use_folds = arg_dict['use_folds'])
    out_dim = datadict['num_classes']
    cur_df = datadict['df']
    label_arr = datadict['label_arr']
    _thresh = datadict['thresh']
    using_toml = datadict['using_toml']
    toml_dict = datadict['toml_dict']
    pl_classdict = datadict['pl_classdict']
    is_classification = datadict['is_classification']

    arg_dict.update({'thresh': _thresh, 'model_type': model_type, 'model_layer_dim': model_layer_dim, 'out_dim': out_dim})

    is_memmap = arg_dict['memmap']
    #### load dataset(s)
    if cur_dsname == "polyrhythms":

        cur_ds = PolyrhythmsData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, classification = is_classification, classdict = pl_classdict, norm_labels = True, layer_idx=arg_dict['layer_idx'], is_64bit = is_64bit, is_memmap = is_memmap, on_share = _on_share)
    elif cur_dsname == 'tempos':
        cur_ds = STHFTempiData(cur_df, embedding_type= arg_dict['embedding_type'], device=device, norm_labels = True, layer_idx= arg_dict['layer_idx'], class_binsize = TEMPOS_CLASS_BINSIZE, num_classes = TP.num_classes, bpm_class_mapper = TP.bpm_class_mapper, is_64bit = is_64bit, is_memmap = is_memmap, on_share = _on_share)
    elif cur_dsname == 'dynamics':
        cur_ds = DynamicsData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], classify_by_subcategory = arg_dict['classify_by_subcategory'], is_64bit = is_64bit, is_memmap = is_memmap, on_share = _on_share)
    elif cur_dsname == 'chords7':
        cur_ds = Chords7Data(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], is_64bit = is_64bit,is_memmap = is_memmap, on_share = _on_share)
    elif cur_dsname == 'chords':
        cur_ds = STHFChordsData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], is_64bit = is_64bit,is_memmap = is_memmap, on_share = _on_share)
    elif cur_dsname == 'time_signatures':
        cur_ds = STHFTimeSignaturesData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], is_64bit = is_64bit,is_memmap = is_memmap, on_share = _on_share)
    elif cur_dsname == 'simple_progressions':
        cur_ds = STHFSimpleProgressionsData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], classify_by_subcategory = arg_dict['classify_by_subcategory'], is_64bit = is_64bit, is_memmap = is_memmap, on_share = _on_share)
        
    elif cur_dsname == 'modemix_chordprog':
        cur_ds = ModemixChordprogData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], classify_by_subcategory = arg_dict['classify_by_subcategory'], is_64bit = is_64bit, is_memmap = is_memmap, on_share = _on_share)
    elif cur_dsname == 'secondary_dominant':
        cur_ds = SecondaryDominantData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], classify_by_subcategory = arg_dict['classify_by_subcategory'], is_64bit = is_64bit, is_memmap = is_memmap, on_share = _on_share)
        


    is_split_debug = arg_dict['split_debug']
    split_debug_name = ''
    _pf = arg_dict['prefix']
    if is_split_debug == True:
        split_annotation = ''
        if is_eval == False:
            split_annotation = 'train'
        else:
            split_annotation = 'eval'
        split_debug_name = f'{_pf}-{cur_dsname}-{split_annotation}'
    cur_subsets = UP.torch_get_train_test_subsets(cur_ds, cur_df, label_arr, train_on_middle = train_on_middle, train_pct = train_pct, test_subpct = test_subpct,seed = seed, debug=is_split_debug, debug_name=split_debug_name, use_folds = arg_dict['use_folds'])
    train_ds = cur_subsets['train']
    valid_ds = cur_subsets['valid']
    test_ds = cur_subsets['test']
    rec_dict = {k:v for (k,v) in arg_dict.items()}
    rec_dict['slurm_job'] = arg_dict['slurm_job']
    arg_dict.update({'train_ds': train_ds, 'valid_ds': valid_ds})

    num_layers = UM.get_embedding_num_layers(emb_type)
    is_single_layer = UM.is_embedding_single_layer(emb_type)
    if arg_dict['debug'] == True and is_eval == False:
        exit()

    study_base_name = f'{args.dataset}-{args.embedding_type}'
    arg_dict['num_layers'] = num_layers
    ### ===== RUNNING PROBE ==== ### 
    if is_eval == False:

        #### running the optuna study
        study_dict = None
        if arg_dict['grid_search'] == True:
     
            search_space = None
            #if param_search == True:
            if is_single_layer == True or force_full_search == True:
                search_space = {'learning_rate_exp': [-5, -4, -3], 'dropout': [0.25, 0.5, 0.75], 'batch_size': [64,256], 'l2_weight_decay_exp': [-4, -3, -2], 'data_norm': [False, True]}
            elif reduced_search == True:
                search_space = {'learning_rate_exp': [-5, -4, -3], 'dropout': [0.25, 0.5, 0.75], 'batch_size': [64], 'l2_weight_decay_exp': [-2], 'data_norm': [True]}
            else:
                search_space = {'learning_rate_exp': [-3], 'dropout': [0.5], 'batch_size': [64], 'l2_weight_decay_exp': [-2], 'data_norm': [True]}
                
            #else:
            #search_space = {'learning_rate_exp': [-5], 'dropout': [0.25], 'l2_weight_decay_exp': [-3]}
            
            if arg_dict['layer_idx']  < 0:
                search_space['layer_idx'] = list(range(num_layers))
            else:
                search_space['layer_idx'] = [arg_dict['layer_idx']]

            study_dict = OU.create_or_load_study(study_base_name, sampler = optuna.samplers.GridSampler(search_space),  maximize = True, prefix=arg_dict['prefix'], script_dir = os.path.dirname(__file__), sampler_dir = 'grid_samplers', db_dir = 'db') 
        else:
            study_dict = OU.create_or_load_study(study_base_name, sampler = optuna.samplers.TPESampler(seed=seed),  maximize = True, prefix=arg_dict['prefix'], script_dir = os.path.dirname(__file__), sampler_dir = 'tpe_samplers', db_dir = 'db') 
        study = study_dict['study']
        study_name = study_dict['study_name']
        study_sampler_path = study_dict['sampler_fpath']
        if using_toml == True:
            flat_toml_dict = UD.flatten_toml_dict(toml_dict)
            rec_dict.update(flat_toml_dict)

        rec_dict['study_name'] = study_name
        UP.record_dict_in_study(study, rec_dict)
        study.set_user_attr('classify_by_subcategory', arg_dict['classify_by_subcategory'])
        study.set_user_attr('thresh', _thresh)
        obj_dict = {k:v for (k,v) in arg_dict.items() if k not in drop_keys}
        objective = partial(_objective, **obj_dict)
        callbacks = [study_callback]

        # init neptune and then run
        to_nep = arg_dict['to_nep'] == True
        num_trials = arg_dict['num_trials']
        nep = None
        nep_callback = None
        nep_id = -1
        if to_nep == True:
            nep, nep_callback = TN.init(param_dict=rec_dict, plots_update_freq = plots_update_freq, log_plot_slice = log_plot_slice, log_plot_contour = log_plot_contour)
            nep_id = nep['sys/id'].fetch()
            callbacks.append(nep_callback)

        if num_trials >= 0:
            study.optimize(objective, timeout = None, n_trials = num_trials, n_jobs=1, gc_after_trial = True, callbacks=callbacks)
        else:
            study.optimize(objective, timeout = None, n_trials = None, n_jobs=1, gc_after_trial = True, callbacks=callbacks)

    else:
        ### ==== JUST EVAL ==== ###
        if arg_dict['debug'] == True:
            exit()

        #### final testing on best trial

        best_param_dict, best_trial_dict, attr_dict  = UB.get_best_params(cur_dsname, arg_dict['embedding_type'], prefix=arg_dict['prefix'])
        # example for dict {'dropout': 0.5, 'l2_weight_decay_exp': -4.0, 'layer_idx': 60.0, 'learning_rate_exp': -5.0}
        dropout = best_param_dict['dropout']['value']
        layer_idx = best_param_dict['layer_idx']['value']
        l2_weight_decay_exp = best_param_dict['l2_weight_decay_exp']['value']
        learning_rate_exp = best_param_dict['learning_rate_exp']['value']
        bs = best_param_dict['batch_size']['value']
        data_norm = best_param_dict['data_norm']['value']
        num_epochs = attr_dict['num_epochs']
        
        best_value = best_trial_dict['value']
        print(f"training probe (valid: {best_value}) with: layer_idx={layer_idx}, dropout={dropout}, lr_exp={learning_rate_exp}, weight_decay_exp={l2_weight_decay_exp}") 
        if len(tomlfile_str) > 0:
            print(f'(toml file: {tomlfile_str})')


        study_name = OU.get_study_name(study_base_name, prefix = arg_dict['prefix'])

        ## model loading and running 
        model = LinearProbe(in_dim=model_layer_dim, hidden_layers = [512],out_dim=out_dim, dropout = dropout, initial_dropout = True)
        held_out_classes = has_held_out_classes(cur_dsname, is_classification)
        
        scaler = None
        if data_norm == True:
            scaler = TST(with_mean = True, with_std = True, dim=model_layer_dim, use_64bit = True, use_constant_feature_mask = True, device = device)

        valid_score = -1.0

        if arg_dict['eval_retrain'] == True:

            valid_score = eval_train(model, scaler = scaler, dataset = cur_dsname, embedding_type = arg_dict['embedding_type'], lr_exp = learning_rate_exp, weight_decay_exp = l2_weight_decay_exp, batch_size = bs, is_classification = is_classification, thresh=_thresh, layer_idx = layer_idx, train_ds = train_ds, valid_ds = valid_ds,  train_on_middle = train_on_middle, classify_by_subcategory = arg_dict['classify_by_subcategory'], model_type=model_type, model_layer_dim=model_layer_dim, out_dim = out_dim, num_epochs=num_dict, prefix = arg_dict['prefix'], early_stopping_check_interval = arg_dict['early_stopping_check_interval'], early_stopping_boredom = arg_dict['early_stopping_boredom'])
            print(f'eval valid score: {valid_score}')
        else:
            trial_number = best_trial_dict['trial_number']
            
            UP.load_probe(model, model_shorthand = emb_type, dataset = cur_dsname, prefix=arg_dict['prefix'], trial_number = trial_number, device = device)

            if data_norm == True:
                UP.load_scaler(scaler, model_shorthand = emb_type, dataset = cur_dsname, prefix=arg_dict['prefix'], trial_number = trial_number, is_64bit = True, device = device)
        test_ds.dataset.set_layer_idx(layer_idx)

        test_loss, test_metrics = valid_test_loop(model,test_ds, loss_fn = None, dataset = cur_dsname, is_classification = is_classification, held_out_classes = held_out_classes, is_testing = True, thresh = _thresh, batch_size = bs, classify_by_subcategory = arg_dict['classify_by_subcategory'], scaler = scaler, file_basename = study_name)
        UP.print_metrics(test_metrics, study_name)
        #UP.save_results_to_study(study, test_metrics)
 
        # some final logging to csv
        rec_dict.update(test_metrics)
        rec_dict['best_trial_obj_value'] = best_value
        
        rec_dict['best_trial_dropout'] = dropout
        rec_dict['best_trial_layer_idx'] = layer_idx
        rec_dict['best_trial_batch_size'] = bs

        rec_dict['eval_valid_score'] = valid_score
        rec_dict['best_lr_exp'] = learning_rate_exp
        rec_dict['best_weight_decay_exp'] = l2_weight_decay_exp
        rec_dict['best_batch_size'] = bs
        rec_dict['best_data_norm'] = data_norm
        rec_dict['num_epochs'] = num_epochs
        test_filt_res = UP.filter_dict(rec_dict, replace_val = 'None', filter_nonstr = True)
        UP.log_results(test_filt_res, study_name)

