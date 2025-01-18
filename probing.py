import torch
from torch import nn
from torch_activations_dataset import STPActivationsData 
from torch_sthf_dataset import STHFTempiData
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
from distutils.util import strtobool
import optuna


### init stuff
torch.manual_seed(5)
shuffle = True
device ='cpu'
if torch.cuda.is_available() == True:
    device = 'cuda'
    torch.cuda.empty_cache()
    torch.set_default_device(device)
tempo_class_binsize=3
polyrhy_clsreg_thresh = 0.01
# hacky way of initialize tempo things with a class_binsize for "classification" from regression
TP.init(tempo_class_binsize)
UP.init(tempo_class_binsize)

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

def train_loop(model, opt_fn, loss_fn, train_dl, is_classification = True):
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

def valid_test_loop(model, loss_fn, eval_dl, dataset = 'polyrhythms', is_classification = True, held_out_classes = False, is_testing = False, poly_regcls_thresh = 0.01):
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
            loss = loss_fn(pred, ground_truth)
            
            cur_truths = ground_truth.cpu().numpy().flatten()
            cur_preds = torch.argmax(pred,axis=1).cpu().numpy().flatten()
            
            if data_idx = 0:
                truths = cur_truths
                preds = cur_preds
            else:
                truths = np.hstack((truths, cur_truths))
                preds = np.hstack((preds, cur_preds))
        else:
            ipt, ground_truth, ground_label = data
            pred = model(ipt.float())
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
        cur_loss = loss.item()
        total_loss += cur_loss
        iters += 1
    # metrics calculating
    metrics = None
    if is_classification == True:
        # only save confusion matrix if testing
        metrics = UP.get_classification_metrics(truths, preds, save_confmat=is_testing)
    else:
        metrics = UP.get_regression_metrics(truths, truth_labels, preds, pred_labels, dataset, held_out_classes = held_out_classes, save_confmat = is_testing)
    
    avg_loss = total_loss/float(iters)
    return avg_loss, metrics

def _objective(trial, dataset = 'polyrhythms', activation_type = 'mg_small_h', is_classification = True, poly_regcls_thresh=0.01):
    # suggested params
    lr = trial.suggest_categorical('learning_rate', [1e-3, 1e-4, 1e-5])
    bs = trial.suggest_categorical('batch_size', [62,256])
    dropout = trial.suggest_categorical('dropout', [0.25, 0.5, 0.75])
    weight_decay = trial.suggest_categorical('l2_weight_decay', [-1, 1e-4, 1e-3])
    num_epochs = trial.suggest_categorical('num_epochs', [50,100,250,500])
    

    model_type = UM.get_model_type(activation_type)  
    model_layer_dim = UM.get_layer_dim(activation_type)
    
    # model init
    out_dim = 1
    if dataset == 'polyrhythms':
        if is_classification == True:
            out_dim = PL.num_poly
    model = Probe(in_dim=model_layer_dim, hidden_layers = [512],out_dim=out_dim, dropout = dropout, initial_dropout = True)

    # optimizer and loss init
    opt_fn = torch.optim.Adam(mode.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = None
    if is_classification == True:
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_fn = nn.MSELoss(reduction='mean')

    # polyrhythm regression has held out classes for "classification"
    held_out_classes = dataset == "polyrhythms" and is_classification == False
    
    for epoch_idx in range(num_epochs):
        train_loss = train_loop(model, opt_fn, loss_fn, train_dl, is_classification = is_classification)
        valid_loss, valid_metrics = valid_test_loop(model, loss_fn, eval_dl, dataset = dataset, is_classification = is_classification, held_out_classes = held_out_classes, is_testing = False, poly_regcls_thresh = poly_regcls_thresh)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-at", "--activation_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-nt", "--num_trials", type=int, default=50, help="number of optuna trials")
    
    args = parser.parse_args()
    arg_dict = vars(args)
    arg_dict.update({'poly_clsreg_thresh': polyrhy_clsreg_thresh})
    cur_time = int(time.time() * 1000)
    study_name = f"{cur_time}-{args.dataset}-{args.activation_type}-{args.num_trials}"
    rdb_string_url = "sqlite:///" + os.path.join(os.path.dirname(__file__), 'db', f'{study_name}.db')
    study = optuna.create_study(study_name=study_name, storage=rdb_string_url, direction='maximize')
    objective = partial(_objective, **arg_dict)
    study.optimize(objective n_trials = arg_dict['num_trials'])
