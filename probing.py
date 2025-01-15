import torch
from torch import nn
from torch_activations_dataset import STPActivationsData 
from torch_sthf_dataset import STHFTempiData
import numpy as np
import torch_nep as TN
import sklearn.metrics as SKM
import os,sys,time,argparse
from functools import partial
import matplotlib.pyplot as plt
import polyrhythms as PL
import util as um
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

def train_classification(trial):
    pass

def _objective(trial, dataset = 'polyrhythms', model_shorthand = 'mg_small_h', is_classification = True):
    # suggested params
    lr = trial.suggest_categorical('learning_rate', [1e-3, 1e-4, 1e-5])
    bs = trial.suggest_categorical('batch_size', [62,256])
    dropout = trial.suggest_categorical('dropout', [0.25, 0.5, 0.75])
    weight_decay = trial.suggest_categorical('l2_weight_decay', [-1, 1e-4, 1e-3])
    num_epochs = trial.suggest_categorical('num_epochs', [50,100,250,500])

    model_type = um.get_model_type(model_shorthand)  
    model_layer_dim = um.get_layer_dim(model_shorthand)
    out_dim = 1
    if dataset == 'polyrhythms':
        if is_classification == True:
            out_dim = PL.num_poly
    model = Probe(in_dim=model_layer_dim, hidden_layers = [512],out_dim=out_dim, dropout = dropout, initial_dropout = True)
    

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
