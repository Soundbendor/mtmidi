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
from torch_polyrhythms_dataset import PolyrhythmsData
from torch_dynamics_dataset import DynamicsData
from torch_modemix_chordprog_dataset import ModemixChordprogData
from torch_secondary_dominant_dataset import SecondaryDominantData
from torch_chords7_dataset import Chords7Data

from hf_tempi_dataset import STHFTempiData
from hf_chords_dataset import STHFChordsData
from hf_timesig_dataset import STHFTimeSignaturesData
from hf_simpleprog_dataset import STHFSimpleProgressionsData

# global declarations (hacky) to save model state dicts
global trial_model_state_dict
global best_model_state_dict
global study_sampler_path



### init stuff
train_pct = 0.7
test_subpct = 0.5
seed = 333
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

### PROBE ###
class Probe(nn.Module):
    def __init__(self, in_dim=4800, hidden_layers = [512],out_dim=10, dropout = 0.5, initial_dropout = True):
        super().__init__()
        self.num_layers = len(hidden_layers)
        self.initial_dropout = initial_dropout

        self.layers = nn.Sequential()
        
        if initial_dropout == True:
            self.layers.append(nn.Dropout(p=dropout))

        # dropout ->
        # num_hidden x (linear -> relu -> dropout) ->
        # linear -> out
        cur_dim = in_dim
        for layer_idx, layer_dim in enumerate(hidden_layers):
            self.layers.append(nn.Linear(cur_dim, layer_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))

            cur_dim = layer_dim
        self.layers.append(nn.Linear(cur_dim, out_dim))

    def forward(self, x):
        return self.layers(x)


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

def train_loop(model, opt_fn, loss_fn, train_ds, batch_size = 64, shuffle = True, is_classification = True):

    train_dl = TUD.DataLoader(train_ds, batch_size = batch_size, shuffle=shuffle, generator=torch.Generator(device=device))
    model.train(True)
    iters = 0
    total_loss = 0
    for data_idx, data in enumerate(train_dl):
        loss = None
        if is_classification == True:
            ipt, ground_truth = data
            pred = model(ipt.float())
            loss = loss_fn(pred, ground_truth)
        else:
            ipt, ground_truth, ground_label = data
            pred = model(ipt.float())
            loss = loss_fn(pred.flatten().float(), ground_truth.flatten().float())
     
        loss.backward()
        opt_fn.step()
        cur_loss = loss.item()
        total_loss += cur_loss
        iters += 1
    avg_loss = total_loss/float(iters)
    return avg_loss

def valid_test_loop(model, eval_ds, loss_fn = None, dataset = 'polyrhythms', is_classification = True, held_out_classes = False, is_testing = False, batch_size = 64, shuffle = True,thresh = 0.01, classify_by_subcategory = False, file_basename=None):
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
            ipt, ground_truth = data
            pred = model(ipt.float())
            if loss_fn != None:
                loss = loss_fn(pred, ground_truth)
            
            cur_truths = ground_truth.detach().cpu().numpy().flatten()
            cur_preds = torch.argmax(pred,axis=1).detach().cpu().numpy().flatten()
            
            if data_idx == 0:
                truths = copy.deepcopy(cur_truths)
                preds = copy.deepcopy(cur_preds)
            else:
                truths = np.hstack((truths, copy.deepcopy(cur_truths)))
                preds = np.hstack((preds, copy.deepcopy(cur_preds)))
        else:
            ipt, ground_truth, ground_label = data
            pred = model(ipt.float())
            if loss_fn != None:
                loss = loss_fn(pred.flatten().float(), ground_truth.flatten().float())
            # stuff for regression "classification"  
            pred_np = pred.detach().cpu().numpy().flatten()
            #if do_regression_classification == True:
            #cur_pred_labels, cur_pred_label_idx = regression_classification(dataset, pred_np, thresh=thresh)
            if data_idx == 0:
                preds = copy.deepcopy(copy.deepcopy(pred_np))
                truths = copy.deepcopy(ground_truth.detach().cpu().numpy().flatten())
                #if do_regression_classification == True:
                #truth_labels = copy.deepcopy(ground_label.detach().cpu().numpy().flatten())
                #pred_labels = copy.deepcopy(cur_pred_label_idx)
            else:
                preds = np.hstack((preds, copy.deepcopy(pred_np)))
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

def _objective(trial, dataset = 'polyrhythms', embedding_type = 'mg_small_h', is_classification = True, thresh=0.01, layer_idx = -1, train_ds = None, valid_ds = None,  train_on_middle = False, classify_by_subcategory = False, model_type='musicgen-small', model_layer_dim=1024, out_dim = 1, prune=False, num_layers = -1, batch_size = 64, num_epochs=250):

    global trial_model_state_dict
    global best_model_state_dict
    model = None
    # suggested params
    lr_exp = trial.suggest_int('learning_rate_exp',-5,-3, step=1)
    lr = 10**lr_exp
    #batch_size_lo = 64
    #batch_size_hi = 256
    #batch_size_step = batch_size_hi - batch_size_lo

    #bs = trial.suggest_int('batch_size', batch_size_lo, batch_size_hi, step=batch_size_step)
    
    #batch_size = bs
    dropout = trial.suggest_float('dropout', 0.25, 0.75, step=0.25)
        
    weight_decay_exp = trial.suggest_int('l2_weight_decay_exp', -4,-2,step=1)
    weight_decay = 10**weight_decay_exp

    lidx = None
    if layer_idx >= 0:
        lidx = layer_idx
    elif num_layers > 0:
        lidx_list = list(range(num_layers))
        lidx = trial.suggest_categorical('layer_idx', lidx_list)
        #lidx = trial.suggest_int('layer_idx', 0, num_layers - 1, step=1)
    else:
        lidx = 0

    train_ds.dataset.set_layer_idx(lidx)
    valid_ds.dataset.set_layer_idx(lidx)

    held_out_classes = has_held_out_classes(dataset, is_classification)     
    model = Probe(in_dim=model_layer_dim, hidden_layers = [512],out_dim=out_dim, dropout = dropout, initial_dropout = True)

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
   
    last_score = None
    for epoch_idx in range(num_epochs):
        train_loss = train_loop(model, opt_fn, loss_fn, train_ds, batch_size = batch_size, is_classification = is_classification)
        valid_loss, valid_metrics = valid_test_loop(model,valid_ds, loss_fn = loss_fn, dataset = dataset, is_classification = is_classification, held_out_classes = held_out_classes, is_testing = False, thresh = thresh, batch_size = batch_size, classify_by_subcategory = classify_by_subcategory)
        cur_score = get_optimization_metric(valid_metrics, is_classification = is_classification)
        # https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html

        if prune == True: 
            if trial.should_prune() == True:
            
                trial.report(cur_score, epoch_idx)
                raise optuna.TrialPruned()
        last_score = cur_score
    #trial.set_user_attr(key='best_state_dict', value=model.state_dict())

    trial_model_state_dict = copy.deepcopy(model.state_dict())
    return last_score
    
# use this to save the best model
# https://stackoverflow.com/questions/62144904/python-how-to-retrieve-the-best-model-from-optuna-lightgbm-study
def study_callback(study, trial):

    global trial_model_state_dict
    global best_model_state_dict

    global study_sampler_path
    with open(study_sampler_path, 'wb') as f:
        pickle.dump(study.sampler, f)
    if study.best_trial.number == trial.number:
        # turns out state dicts are not json serializable (so doesn't work)
        #trial.set_user_attr(key='best_state_dict', value=copy.deepcopy(trial.user_attrs['best_state_dict']))
        best_model_state_dict = copy.deepcopy(trial_model_state_dict)


# training for evaluation
def eval_train(model, dataset = 'polyrhythms', embedding_type = 'mg_small_h', is_classification = True, thresh=0.01, layer_idx = 0, train_ds = None, valid_ds = None,  train_on_middle = False, classify_by_subcategory = False, model_type='musicgen-small', lr_exp = -3, weight_decay_exp = -3,  model_layer_dim=1024, out_dim = 1, batch_size = 64, num_epochs=250):

    lr = 10**lr_exp

        
    weight_decay = 10**weight_decay_exp

    train_ds.dataset.set_layer_idx(layer_idx)
    valid_ds.dataset.set_layer_idx(layer_idx)

    held_out_classes = has_held_out_classes(dataset, is_classification)     
    model = Probe(in_dim=model_layer_dim, hidden_layers = [512],out_dim=out_dim, dropout = dropout, initial_dropout = True)

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
   
    last_score = None
    for epoch_idx in range(num_epochs):
        train_loss = train_loop(model, opt_fn, loss_fn, train_ds, batch_size = batch_size, is_classification = is_classification)
        valid_loss, valid_metrics = valid_test_loop(model,valid_ds, loss_fn = loss_fn, dataset = dataset, is_classification = is_classification, held_out_classes = held_out_classes, is_testing = False, thresh = thresh, batch_size = batch_size, classify_by_subcategory = classify_by_subcategory)
        cur_score = get_optimization_metric(valid_metrics, is_classification = is_classification)

        last_score = cur_score

    return last_score



if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-et", "--embedding_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-nt", "--num_trials", type=int, default=3000, help="number of optuna trials")
    parser.add_argument("-li", "--layer_idx", type=int, default=-1, help="< 0 to optimize by optuna, else specifies layer_idx 0-indexed")
    parser.add_argument("-cls", "--is_classification", type=strtobool, default=True, help="is classification")
    parser.add_argument("-tom", "--train_on_middle", type=strtobool, default=False, help="train on middle")
    parser.add_argument("-rc", "--do_regression_classification", type=strtobool, default=False, help="do regression classification")
    parser.add_argument("-nep", "--to_nep", type=strtobool, default=True, help="log on neptune")
    parser.add_argument("-cbs", "--classify_by_subcategory", type=strtobool, default=False, help="classify by subcategory for dynamics, by progression for chord progression datasets")
    parser.add_argument("-pf", "--prefix", type=int, default=-1, help="specify a prefix > 0 for save files (db, etc.) for potential reloading (if file exists)")
    parser.add_argument("-tf", "--toml_file", type=str, default="", help="toml file in toml directory with exclude category listing vals to exclude by col, amongst other settings")
    parser.add_argument("-ev", "--eval", type=strtobool, default=False, help="evalute on best performing params recorded")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="hacky way of syntax debugging")
    parser.add_argument("-epc", "--num_epochs", type=int, default=250, help="number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-pr", "--prune", type=strtobool, default=True, help="do pruning")
    parser.add_argument("-gr", "--grid_search", type=strtobool, default=False, help="grid search")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="load embeddings as memmap, else npy")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")

    # obj_dict is for passing to objective function, is arg_dict without drop_keys
    # rec_dict is for passing to neptune and study (has drop keys)
    # arg_dict just has everything
    drop_keys = set(['to_nep', 'num_trials', 'toml_file', 'do_regression_classification', 'debug', 'memmap', 'slurm_job','grid_search', 'prefix','eval'])
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
    
    _classify_by_subcategory = arg_dict['classify_by_subcategory'] 

    

    datadict  = UD.load_data_dict(cur_dsname, classify_by_subcategory = _classify_by_subcategory, tomlfile_str = tomlfile_str)
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

        cur_ds = PolyrhythmsData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, classification = is_classification, classdict = pl_classdict, norm_labels = True, layer_idx=arg_dict['layer_idx'], is_64bit = is_64bit, is_memmap = is_memmap)
    elif cur_dsname == 'tempos':
        cur_ds = STHFTempiData(cur_df, embedding_type= arg_dict['embedding_type'], device=device, norm_labels = True, layer_idx= arg_dict['layer_idx'], class_binsize = TEMPOS_CLASS_BINSIZE, num_classes = TP.num_classes, bpm_class_mapper = TP.bpm_class_mapper, is_64bit = is_64bit, is_memmap = is_memmap)
    elif cur_dsname == 'dynamics':
        cur_ds = DynamicsData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], classify_by_subcategory = arg_dict['classify_by_subcategory'], is_64bit = is_64bit, is_memmap = is_memmap)
    elif cur_dsname == 'chords7':
        cur_ds = Chords7Data(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], is_64bit = is_64bit,is_memmap = is_memmap)
    elif cur_dsname == 'chords':
        cur_ds = STHFChordsData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], is_64bit = is_64bit,is_memmap = is_memmap)
    elif cur_dsname == 'time_signatures':
        cur_ds = STHFTimeSignaturesData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], is_64bit = is_64bit,is_memmap = is_memmap)
    elif cur_dsname == 'simple_progressions':
        cur_ds = STHFSimpleProgressionsData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], classify_by_subcategory = arg_dict['classify_by_subcategory'], is_64bit = is_64bit, is_memmap = is_memmap)
        
    elif cur_dsname == 'modemix_chordprog':
        cur_ds = ModemixChordprogData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], classify_by_subcategory = arg_dict['classify_by_subcategory'], is_64bit = is_64bit, is_memmap = is_memmap)
    elif cur_dsname == 'secondary_dominant':
        cur_ds = SecondaryDominantData(cur_df, embedding_type = arg_dict['embedding_type'], device=device, layer_idx=arg_dict['layer_idx'], classify_by_subcategory = arg_dict['classify_by_subcategory'], is_64bit = is_64bit, is_memmap = is_memmap)
        


    cur_subsets = UP.torch_get_train_test_subsets(cur_ds, label_arr, train_on_middle = train_on_middle, train_pct = train_pct, test_subpct = test_subpct,seed = seed)
    train_ds = cur_subsets['train']
    valid_ds = cur_subsets['valid']
    test_ds = cur_subsets['test']
    rec_dict = {k:v for (k,v) in arg_dict.items()}
    rec_dict['slurm_job'] = arg_dict['slurm_job']
    arg_dict.update({'train_ds': train_ds, 'valid_ds': valid_ds})

    num_layers = UM.get_embedding_num_layers(emb_type)

    if arg_dict['debug'] == True and is_eval == False:
        exit()

    study_base_name = f'{args.dataset}-{args.embedding_type}'
    ### ===== RUNNING PROBE ==== ### 
    if is_eval == False:

        #### running the optuna study
        study_dict = None
        if arg_dict['grid_search'] == True:
     
            search_space = None
            if param_search == True:
                search_space = {'learning_rate_exp': [-5, -4, -3], 'dropout': [0.25, 0.5, 0.75], 'l2_weight_decay_exp': [-4, -3, -2]}
            else:
                search_space = {'learning_rate_exp': [-5], 'dropout': [0.25], 'l2_weight_decay_exp': [-3]}
            
            if arg_dict['layer_idx']  < 0:
                search_space['layer_idx'] = list(range(num_layers))
            else:
                search_space['layer_idx'] = [arg_dict['layer_idx']]

            study_dict = OU.create_or_load_study(study_base_name, sampler = optuna.samplers.GridSampler(search_space),  maximize = True, prefix=arg_dict['prefix'], script_dir = os.path.dirname(__file__), sampler_dir = 'grid_samplers', db_dir = 'db') 
        else:
            arg_dict['num_layers'] = num_layers
            study_dict = OU.create_or_load_study(study_base_name, sampler = optuna.samplers.TPESampler(),  maximize = True, prefix=arg_dict['prefix'], script_dir = os.path.dirname(__file__), sampler_dir = 'tpe_samplers', db_dir = 'db') 
        study = study_dict['study']
        study_name = study_dict['study_name']
        study_sampler_path = study_dict['sampler_fpath']
        if using_toml == True:
            flat_toml_dict = UD.flatten_toml_dict(toml_dict)
            rec_dict.update(flat_toml_dict)
            UP.record_dict_in_study(study, rec_dict)

        rec_dict['study_name'] = study_name
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

        #### final testing on best trial
        dropout = study.best_params.get('dropout', 0.5)
        layer_idx = arg_dict.get('layer_idx', 0)
        #bs = study.best_params.get('batch_size', 64)
        bs = arg_dict['batch_size']
        best_value = study.best_value


        if user_specify_layer_idx == False:
            layer_idx = study.best_params.get('layer_idx', 0)

        ## model loading and running 
        model = Probe(in_dim=model_layer_dim, hidden_layers = [512],out_dim=out_dim, dropout = dropout, initial_dropout = True)
        model.load_state_dict(best_model_state_dict)
        held_out_classes = has_held_out_classes(cur_dsname, is_classification)
        test_ds.dataset.set_layer_idx(layer_idx)
        test_loss, test_metrics = valid_test_loop(model, test_ds, loss_fn = None, dataset = cur_dsname, is_classification = is_classification, held_out_classes = held_out_classes, is_testing = True,  thresh = arg_dict['thresh'], classify_by_subcategory = arg_dict['classify_by_subcategory'], batch_size = bs, file_basename = study_name)
        UP.print_metrics(test_metrics, study_name)
        UP.save_results_to_study(study, test_metrics)

        # some final logging to neptune
        test_filt_nep = UP.filter_dict(test_metrics, replace_val = None, filter_nonstr = False)
        if to_nep == True:
            UP.neptune_log(nep, test_filt_nep)
            TN.tidy(study, nep)

        # some final logging to csv
        rec_dict.update(test_metrics)
        rec_dict['best_trial_obj_value'] = best_value
        
        rec_dict['best_trial_dropout'] = dropout
        rec_dict['best_trial_layer_idx'] = layer_idx
        #rec_dict['best_trial_batch_size'] = bs

        rec_dict['best_lr_exp'] = study.best_params['learning_rate_exp']
        rec_dict['best_weight_decay_exp'] = study.best_params['l2_weight_decay_exp']
        test_filt_res = UP.filter_dict(rec_dict, replace_val = 'None', filter_nonstr = True)
        UP.log_results(test_filt_res, study_name)

    else:
        ### ==== JUST EVAL ==== ###
        if arg_dict['debug'] == True:
            exit()

        #### final testing on best trial

        best_param_dict, best_trial, best_value = UB.get_best_params(cur_dsname, arg_dict['embedding_type'], prefix=arg_dict['prefix'])
        # example for dict {'dropout': 0.5, 'l2_weight_decay_exp': -4.0, 'layer_idx': 60.0, 'learning_rate_exp': -5.0}
        dropout = best_param_dict.get('dropout', 0.5)
        layer_idx = int(best_param_dict.get('layer_idx', 0))
        l2_weight_decay_exp = best_param_dict.get('l2_weight_decay_exp', -4.0)

        learning_rate_exp = int(best_param_dict.get('learning_rate_exp', 0))

        print(f"training probe with: layer_idx={layer_idx}, dropout={dropout}, lr_exp={learning_rate_exp}, weight_decay_exp={l2_weight_decay_exp}") 
        #bs = study.best_params.get('batch_size', 64)
        #bs = arg_dict['batch_size']
        bs = 64 # batch size used
        num_epochs = 250 # num_epochs used


        study_name = OU.get_study_name(study_base_name, prefix = arg_dict['prefix'])

        ## model loading and running 
        model = Probe(in_dim=model_layer_dim, hidden_layers = [512],out_dim=out_dim, dropout = dropout, initial_dropout = True)
        held_out_classes = has_held_out_classes(cur_dsname, is_classification)

        eval_train(model, dataset = 'polyrhythms', embedding_type = arg_dict['embedding_type'], is_classification = is_classification, thresh=_thresh, layer_idx = layer_idx, train_ds = train_ds, valid_ds = valid_ds,  train_on_middle = train_on_middle, classify_by_subcategory = False, lr_exp = learning_rate_exp, weight_decay_exp = l2_weight_decay_exp, model_type=model_type, model_layer_dim=model_layer_dim, out_dim = out_dim, batch_size = bs, num_epochs=num_epochs)
        test_ds.dataset.set_layer_idx(layer_idx)
        test_loss, test_metrics = valid_test_loop(model, test_ds, loss_fn = None, dataset = cur_dsname, is_classification = is_classification, held_out_classes = held_out_classes, is_testing = True,  thresh = arg_dict['thresh'], classify_by_subcategory = arg_dict['classify_by_subcategory'], batch_size = bs, file_basename = study_name)
        UP.print_metrics(test_metrics, study_name)
        UP.save_results_to_study(study, test_metrics)
 
        # some final logging to csv
        rec_dict.update(test_metrics)
        rec_dict['best_trial_obj_value'] = best_value
        
        rec_dict['best_trial_dropout'] = dropout
        rec_dict['best_trial_layer_idx'] = layer_idx
        #rec_dict['best_trial_batch_size'] = bs

        rec_dict['best_lr_exp'] = learning_rate_exp
        rec_dict['best_weight_decay_exp'] = l2_weight_decay_exp
        test_filt_res = UP.filter_dict(rec_dict, replace_val = 'None', filter_nonstr = True)
        UP.log_results(test_filt_res, study_name)

