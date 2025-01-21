import torch
from torch import nn
import torch.utils.data as TUD
from hf_tempi_dataset import STHFTempiData
import numpy as np
import torch_nep as TN
import sklearn.metrics as SKM
import os,sys,time,argparse
from functools import partial
import polyrhythms as PL
import util as UM
import utils_probing as UP
import tempi as TP
import tomllib
from torch_polyrhythms_dataset import PolyrhythmsData
from distutils.util import strtobool
import optuna
import copy


### some params to play around with
# polyrhythms
exclude_polys = []
exclude_offset_lvls = []

### init stuff
is_64bit = True # if embeddings are 64 bit
train_pct = 0.7
test_subpct = 0.5
seed = 5
torch.manual_seed(seed)
shuffle = True
# neptune stuff
plots_update_freq = 10
log_plot_slice = True
log_plot_contour = True
# hacky way of initialize tempo things with a class_binsize for "classification" from regression
TEMPOS_CLASS_BINSIZE=3
POLY_REGCLS_THRESH = 0.01

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
        self.num_layers = len(hidden_layers)
        cur_dim = in_dim
        self.initial_dropout = initial_dropout

        self.layers = nn.Sequential()
        
        if initial_dropout == True:
            self.layers.append(nn.Dropout(p=dropout))

        # dropout ->
        # num_hidden x (linear -> relu -> dropout) ->
        # linear -> out
        for layer_idx, layer_dim in enumerate(hidden_layers):
            self.layers.append(nn.Linear(cur_dim, layer_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))

            cur_dim = layer_dim
        self.layers.append(nn.Linear(cur_dim, out_dim))

    def forward(self, x):
        return self.layers(x)


### REGRESSION "CLASSIFICATION"
def regression_classification(dataset, predictions, poly_regcls_thresh=0.01):
    # predictions should be in numpy format
    cur_pred_labels = None
    cur_pred_label_idx = None
    if dataset == 'polyrhythms':
        cur_pred_labels = [PL.get_nearest_poly(x, thresh=poly_regcls_thresh) for x in predictions]
        cur_pred_label_idx = np.array([PL.reg_polystr_to_idx[x] for x in cur_pred_labels])
    elif dataset == 'tempi':
        cur_pred_labels = [TP.get_nearest_bpmclass(x, classlist_sorted, thresh=thresh) for x in predictions]
        #print('test2', pred_np.shape, pred_np.dtype, len(cur_pred_labels))
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
            pred = model(ipt)
            loss = loss_fn(pred, ground_truth)
        else:
            ipt, ground_truth, ground_label = data
            pred = model(ipt.float())
            loss = loss_fn(pred.flatten().float(), ground_truth.flatten().float(), reduction = 'mean')
     
        loss.backward()
        opt_fn.step()
        cur_loss = loss.item()
        total_loss += cur_loss
        iters += 1
    avg_loss = total_loss/float(iters)
    return avg_loss

def valid_test_loop(model, eval_ds, loss_fn = None, dataset = 'polyrhythms', is_classification = True, held_out_classes = False, is_testing = False, batch_size = 64, shuffle = True, poly_regcls_thresh = 0.01):
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
            pred = model(ipt)
            if loss_fn != None:
                loss = loss_fn(pred, ground_truth)
            
            cur_truths = ground_truth.cpu().numpy().flatten()
            cur_preds = torch.argmax(pred,axis=1).cpu().numpy().flatten()
            
            if data_idx == 0:
                truths = cur_truths
                preds = cur_preds
            else:
                truths = np.hstack((truths, cur_truths))
                preds = np.hstack((preds, cur_preds))
        else:
            ipt, ground_truth, ground_label = data
            pred = model(ipt.float())
            if loss_fn != None:
                loss = loss_fn(pred.flatten().float(), ground_truth.flatten().float(), reduction = 'mean')
            # stuff for regression "classification"  
            pred_np = pred.cpu().numpy().flatten()
            cur_pred_labels, cur_pred_label_idx = regression_classification(dataset, pred_np, poly_regcls_thresh=poly_regcls_thresh)
            if data_idx == 0:
                preds = pred_np
                truths = ground_truth.cpu().numpy().flatten()
                truth_labels = ground_label.cpu().numpy().flatten()
                pred_labels = cur_pred_label_idx
            else:
                preds = np.hstack((preds, pred_np))
                truths = np.hstack((truths, ground_truth.cpu().numpy().flatten()))
                truth_labels = np.hstack((truth_labels, ground_label.cpu().numpy().flatten()))
                pred_labels = np.hstack((pred_labels, cur_pred_label_idx))

        # loss bookkeeping
        if loss_fn != None:
            cur_loss = loss.item()
            total_loss += cur_loss
            iters += 1
    # metrics calculating
    metrics = None
    if is_classification == True:
        # only save confusion matrix if testing
        metrics = UP.get_classification_metrics(truths, preds, dataset, save_confmat=is_testing)
    else:
        metrics = UP.get_regression_metrics(truths, truth_labels, preds, pred_labels, dataset, held_out_classes = held_out_classes, save_confmat = is_testing)
    
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
    return (dataset in ["polyrhythms", "tempos"]) and is_classification == False

def _objective(trial, dataset = 'polyrhythms', embedding_type = 'mg_small_h', is_classification = True, poly_regcls_thresh=0.01, layer_idx = -1, train_ds = None, valid_ds = None,  train_on_middle = False, model_type='musicgen-small', model_layer_dim=1024, out_dim = 1, to_nep = True):
    # suggested params
    lr = trial.suggest_categorical('learning_rate', [1e-3, 1e-4, 1e-5])
    bs = trial.suggest_categorical('batch_size', [62,256])
    dropout = trial.suggest_categorical('dropout', [0.25, 0.5, 0.75])
    weight_decay = trial.suggest_categorical('l2_weight_decay', [-1, 1e-4, 1e-3])
    num_epochs = trial.suggest_categorical('num_epochs', [50,100,250,500])
    user_specify_layer_idx = layer_idx >= 0 
    lidx = None
    # -1 since 0-indexed
    max_layer_idx = UM.get_embedding_num_layers(embedding_type) - 1
    if user_specify_layer_idx == True:
        lidx = min(max_layer_idx, layer_idx)
    else:
        lidx = trial.suggest_int('layer_idx', 0, max_layer_idx, step=1)

    train_ds.set_layer_idx(lidx)
    valid_ds.set_layer_idx(lidx)

    held_out_classes = has_held_out_classes(dataset, is_classification)     
    model = Probe(in_dim=model_layer_dim, hidden_layers = [512],out_dim=out_dim, dropout = dropout, initial_dropout = True)

    # optimizer and loss init
    opt_fn = None
    if weight_decay > 0:
        opt_fn = torch.optim.Adam(mode.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt_fn = torch.optim.Adam(mode.parameters(), lr=lr)
    loss_fn = None
    if is_classification == True:
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_fn = nn.MSELoss(reduction='mean')

    # polyrhythm and tempi regression has held out classes for "classification"
    held_out_classes = (dataset in ["polyrhythms", "tempos"]) and is_classification == False
   
    last_score = None
    for epoch_idx in range(num_epochs):
        train_loss = train_loop(model, opt_fn, loss_fn, train_ds, batch_size = bs, is_classification = is_classification)
        valid_loss, valid_metrics = valid_test_loop(model,valid_ds, loss_fn = loss_fn, dataset = dataset, is_classification = is_classification, held_out_classes = held_out_classes, is_testing = False, poly_regcls_thresh = poly_regcls_thresh, batch_size = bs)
        cur_score = get_optimization_metric(valid_metrics, is_classification = is_classification)
        # https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html
        trial.report(cur_score, epoch_idx)

        if trial.should_prune() == True:
            raise optuna.TrialPruned()
        last_score = cur_score
    trial.set_user_attr(key='best_state_dict', value=model.state_dict())
    return last_score
    
# use this to save the best model
# https://stackoverflow.com/questions/62144904/python-how-to-retrieve-the-best-model-from-optuna-lightgbm-study
def study_callback(study, trial):
    if study.best_trial_number == trial.number:
        # deepcopy since probably state dict is a reference
        trial.set_user_attr(key='best_state_dict', value=copy.deepcopy(trial.user_attrs['best_state_dict']))

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-et", "--embedding_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-nt", "--num_trials", type=int, default=50, help="number of optuna trials")
    parser.add_argument("-li", "--layer_idx", type=int, default=-1, help="< 0 to optimize by optuna, else specifies layer_idx 0-indexed")
    parser.add_argument("-cls", "--is_classification", type=strtobool, default=True, help="is classification")
    parser.add_argument("-tom", "--train_on_middle", type=strtobool, default=True, help="train on middle")
    parser.add_argument("-nep", "--to_nep", type=strtobool, default=True, help="log on neptune")
   
    #### some more logic to define experiments
    args = parser.parse_args()
    arg_dict = vars(args)
    # model type is slightly distinct from embedding_type (which is also shorthand) because musicgen-encoder uses musicgen-large
    model_type = UM.get_model_type(arg_dict['embedding_type'])  
    model_layer_dim = UM.get_layer_dim(arg_dict['embedding_type'])
    
    out_dim = 1
    if arg_dict['dataset'] == 'polyrhythms':
        if arg_dict['is_classification'] == True:
            out_dim = PL.num_poly

    arg_dict.update({'poly_regcls_thresh': POLY_REGCLS_THRESH, 'model_type': model_type, 'model_layer_dim': model_layer_dim, 'out_dim': out_dim})

    #### some variable definitions
    cur_ds = None
    label_arr = None
    train_on_middle = False 
    user_specify_layer_idx = arg_dict['layer_idx'] >= 0 
    #### load dataset(s)
    if arg_dict['dataset'] == "polyrhythms":
       cur_ds = PolyrhythmsData(embedding_type = arg_dict['embedding_type'], device=device, classification = arg_dict["is_classification"], exclude_polys = exclude_polys, exclude_offset_lvls = exclude_offset_lvls, norm_labels = True, layer_idx=arg_dict['layer_idx'], is_64bit = is_64bit)
       label_arr = cur_ds.all_pstr
    elif arg_dict['dataset'] == 'tempos':
       cur_ds = STHFTempiData(embedding_type= arg_dict['embedding_type'], device=device, norm_labels = True, layer_idx= arg_dict['layer_idx'], class_binsize = TEMPOS_CLASS_BINSIZE, num_classes = TP.num_classes, bpm_class_mapper = TP.bpm_class_mapper, is_64bit = is_64bit)
       label_arr = cur_ds.all_classes
    train_ds, valid_ds, test_ds = UP.get_train_valid_test_subsets(cur_ds, label_all, train_on_middle = arg_dict['train_on_middle'], train_pct = train_pct, test_subpct = test_subpct, seed = seed)
    arg_dict.update({'train_ds': train_ds, 'valid_ds': valid_ds})

    #### running the optuna study
    cur_time = int(time.time() * 1000)
    study_name = f"{cur_time}-{args.dataset}-{args.embedding_type}-{args.num_trials}"
    rdb_string_url = "sqlite:///" + os.path.join(os.path.dirname(__file__), 'db', f'{study_name}.db')

    study = optuna.create_study(study_name=study_name, storage=rdb_string_url, direction='maximize')
    objective = partial(_objective, **arg_dict)
    callbacks = [study_callback]

    # init neptune and then run
    nep = None
    nep_callback = None
    if arg_dict['to_nep'] == True:
        nep, nep_callback = TN.init(arg_dict, plots_update_freq = plots_update_freq, log_plot_slice = log_plot_slice, log_plot_contour = log_plot_contour)
        callbacks.append(nep_callback)

    study.optimize(objective, n_trials = arg_dict['num_trials'], callbacks=[study_callback, ])

    #### final testing on best trial
    best_state_dict = study.user_attrs['best_state_dict']
    dropout = study.best_params['dropout']
    layer_idx = arg_dict['layer_idx']
    bs = arg_dict['batch_size']
    if user_specify_layer_idx == False:
        layer_idx = study.best_params['layer_idx']

    ## model loading and running 
    model = Probe(in_dim=model_layer_dim, hidden_layers = [512],out_dim=out_dim, dropout = dropout, initial_dropout = True)
    model.load_state_dict(best_state_dict)
    held_out_classes = has_held_out_classes(dataset, is_classification)
    test_ds.set_layer_idx(layer_idx)
    test_loss, test_metrics = valid_test_loop(model, test_ds, loss_fn = None, dataset = arg_dict['dataset'], is_classification = arg_dict['is_classification'], held_out_classes = held_out_classes, is_testing = True, poly_regcls_thresh = arg_dict['poly_regcls_thresh'], batch_size = bs)
    UP.print_metrics(test_metrics, study_name)
    if arg_dict['to_nep'] == True:
        UP.neptune_log(nep, test_metrics)
        TN.tidy(study, nep)


