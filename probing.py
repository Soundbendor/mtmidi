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

def valid_test_loop(model, loss_fn, eval_dl, dataset = 'polyrhythms', is_classification = True):
    model.eval()
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

def _objective(trial, dataset = 'polyrhythms', model_shorthand = 'mg_small_h', is_classification = True):
    # suggested params
    lr = trial.suggest_categorical('learning_rate', [1e-3, 1e-4, 1e-5])
    bs = trial.suggest_categorical('batch_size', [62,256])
    dropout = trial.suggest_categorical('dropout', [0.25, 0.5, 0.75])
    weight_decay = trial.suggest_categorical('l2_weight_decay', [-1, 1e-4, 1e-3])
    num_epochs = trial.suggest_categorical('num_epochs', [50,100,250,500])

    model_type = UM.get_model_type(model_shorthand)  
    model_layer_dim = UM.get_layer_dim(model_shorthand)
    
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

    
    for epoch_idx in range(num_epochs):
        train_loss = train_loop(model, opt_fn, loss_fn, train_dl, is_classification = is_classification)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-at", "--activation_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-nt", "--num_trials", type=int, default=50, help="number of optuna trials")
    
    args = parser.parse_args()

    cur_time = int(time.time() * 1000)
    study_name = f"{cur_time}-{args.dataset}-{args.activation_type}-{args.num_trials}"
    rdb_string_url = "sqlite:///" + os.path.join(os.path.dirname(__file__), 'db', f'{study_name}.db')
    study = optuna.create_study(study_name=study_name, storage=rdb_string_url, direction='maximize')
    objective = partial(_objective, args)
    study.optimize(objective, n_trials=args.num_trials)
