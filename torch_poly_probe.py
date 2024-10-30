import torch
from torch import nn
from torch_activations_dataset import STPActivationsData 
from torch_probe_model import LinearProbe
import numpy as np
import neptune
import os

classdict = {'2a3': 0, '3a4': 1, '3a5': 2, '4a5': 3, '5a6': 4, '5a7': 5, '6a7': 6, '7a8': 7}
rev_classdict = {i:x for (x,i) in classdict.items()}
num_classes = len(classdict)

user_folder = os.path.expanduser("~" + os.getenv("USER")) 
data_folder = os.path.join(user_folder, "ds", "jukebox_acts_36")
params = {'batch_size': 16, 'num_epochs': 100, 'lr': 1e-3, 'dropout': 0.5, 'hidden_layers': [512], 'use_cuda': True}
bs = params['batch_size']
device ='cpu'
lr = params['lr']
num_epochs = params['num_epochs']
dropout = params['dropout']
hidden_layers = params['hidden_layers']

if torch.cuda.is_available() == True and params["use_cuda"] == True:
    device = 'cuda'
    torch.cuda.empty_cache()
    torch.set_default_device(device)


train_data = STPActivationsData(device=device, data_folder = data_folder, classdict = classdict, num_classes = num_classes, set_type='train')
valid_data = STPActivationsData(device=device, data_folder = data_folder, classdict = classdict, num_classes = num_classes, set_type='valid')
test_data = STPActivationsData(device=device, data_folder = data_folder, classdict = classdict, num_classes = num_classes, set_type='test')

model = LinearProbe(num_classes=num_classes,hidden_layers = hidden_layers, dropout = dropout).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

def train_epoch(_model, _tdl, _lossfn, _optimfn):
    _model.train(True)
    tot_loss = 0.
    for didx, data in enumerate(_tdl):
        ipt, ground_truth = data
        _optimfn.zero_grad()
        pred = _model(ipt)
        _loss = _lossfn(pred, ground_truth)
        _loss.backward()
        _optimfn.step()
        tot_loss += _loss.item()

    return tot_loss

def valid_epoch(_model, _vdl, _lossfn):
    _model.eval()
    tot_loss = 0.
    with torch.no_grad():
        for didx, data in enumerate(_vdl):
            ipt, ground_truth = data
            pred = _model(ipt)
            _loss = _lossfn(pred, ground_truth)
            tot_loss += _loss.item()
    return tot_loss


def train_valid(_model, _traindata, _validdata, _lossfn, _optimfn, batch_size=16, epochs=10, _nep=None):
    train_losses = []
    valid_losses = []
    to_nep = _nep != None
    for eidx in range(epochs):
        train_dl = torch.utils.data.DataLoader(_traindata, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
        valid_dl = torch.utils.data.DataLoader(_validdata, batch_size = batch_size, shuffle=True, generator=torch.Generator(device=device))
        train_loss = train_epoch(_model, train_dl, _lossfn, _optimfn)
        train_losses.append(train_loss)
        valid_loss = valid_epoch(_model, train_dl, _lossfn)
        valid_losses.append(valid_loss)
        if to_nep == True:
            _nep['train/loss'].append(train_loss)
            _nep['valid/loss'].append(valid_loss)
    return train_losses, valid_losses

def test(_model, _testdata, _lossfn, batch_size = 16, _nep=None)
    to_nep = _nep != None
    _model.eval()
    tot_loss = 0.

    with torch.no_grad():
        for didx, data in enumerate(_vdl):
            ipt, ground_truth = data
            pred = _model(ipt)
            _loss = _lossfn(pred, ground_truth)
            tot_loss += _loss.item()

