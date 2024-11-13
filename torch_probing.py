import torch
from torch import nn
from torch_activations_dataset import STPActivationsData 
from torch_probe_model import LinearProbe
import numpy as np
import torch_nep as TN
import sklearn.metrics as SKM
import os
import matplotlib.pyplot as plt
import time
import sys

torch.manual_seed(3)
classdict = {'2a3': 0, '3a4': 1, '3a5': 2, '4a5': 3, '5a6': 4, '5a7': 5, '6a7': 6, '7a8': 7}
rev_classdict = {i:x for (x,i) in classdict.items()}
class_arr = [k for (k,v) in classdict.items()]
num_classes = len(classdict)

data_debug = False
to_nep = True
split = 2
classification = False
bs = 256
lr = 1e-3
num_epochs = 500
if classification == False:
    num_epochs = 100
    lr = 1e-4
dropout = 0.5
#hidden_layers = [512]
hidden_layers = []


res_dir = "res"
user_folder = os.path.expanduser("~" + os.getenv("USER")) 
#data_folder = os.path.join(user_folder, "ds", "jukebox_acts_36")
data_folder = os.path.join("acts", "jukebox_acts_36")
params = {'batch_size': bs, 'num_epochs': num_epochs, 'lr': lr, 'dropout': dropout, 'use_cuda': True, 'split': split, 'classification': classification}
device ='cpu'

csvfile = os.path.join('csv', 'polyrhy_split1.csv')
if params['split'] != 1:
    print('running split 2')
    csvfile = os.path.join('csv', 'polyrhy_split2.csv')

if torch.cuda.is_available() == True and params["use_cuda"] == True:
    device = 'cuda'
    torch.cuda.empty_cache()
    torch.set_default_device(device)


train_data = STPActivationsData(csvfile = csvfile, device=device, data_folder = data_folder, classdict = classdict, num_classes = num_classes, set_type='train', classification = classification)
valid_data = STPActivationsData(csvfile = csvfile,  device=device, data_folder = data_folder, classdict = classdict, num_classes = num_classes, set_type='val', classification = classification)
test_data = STPActivationsData(csvfile = csvfile,  device=device, data_folder = data_folder, classdict = classdict, num_classes = num_classes, set_type='test', classification = classification)

if data_debug == True:
    num_train = len(train_data)
    num_valid = len(valid_data)
    num_test = len(test_data)
    print(f"num_train: {num_train}, num_valid: {num_valid}, num_test: {num_test}")
    sys.exit()
model = None
loss_fn = None
if classification == True:
    model = LinearProbe(num_classes=num_classes,hidden_layers = hidden_layers, dropout = dropout).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
else:
    model = LinearProbe(num_classes=1,hidden_layers = hidden_layers, dropout = dropout).double().to(device)
    loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

def train_epoch(_model, _tdl, _lossfn, _optimfn):
    _model.train(True)
    tot_loss = 0.
    for didx, data in enumerate(_tdl):
        ipt, ground_truth = data
        _optimfn.zero_grad()
        _loss = None
        if classification == True:
            pred = _model(ipt)
            _loss = _lossfn(pred, ground_truth)
        else:
            pred = _model(ipt.double())
            _loss = _lossfn(pred.flatten(), ground_truth)
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
            if classification == True:
                pred = _model(ipt)
                _loss = _lossfn(pred, ground_truth)
            else:
                pred = _model(ipt.double())
                _loss = _lossfn(pred.flatten(), ground_truth)

            tot_loss += _loss.item()
    return tot_loss


def train_valid(_model, _traindata, _validdata, _lossfn, _optimfn, batch_size=16, epochs=10, _nep=None):
    train_losses = []
    valid_losses = []
    to_nep = _nep != None
    print("--- training ---")
    for eidx in range(epochs):
        print(f'Epoch {eidx+1}')
        train_dl = torch.utils.data.DataLoader(_traindata, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
        valid_dl = torch.utils.data.DataLoader(_validdata, batch_size = batch_size, shuffle=True, generator=torch.Generator(device=device))
        train_loss = train_epoch(_model, train_dl, _lossfn, _optimfn)
        train_losses.append(train_loss)
        valid_loss = valid_epoch(_model, train_dl, _lossfn)
        valid_losses.append(valid_loss)
        print(f'train: {train_loss}, valid: {valid_loss}')
        if to_nep == True:
            _nep['train/loss'].append(train_loss)
            _nep['valid/loss'].append(valid_loss)
    return train_losses, valid_losses



def test_regression(_model, _testdata, batch_size = 16, _nep=None):
    _tdl = torch.utils.data.DataLoader(_testdata, batch_size = batch_size, shuffle=True, generator=torch.Generator(device=device))
    to_nep = _nep != None
    _model.eval()
    tot_loss = 0.
    preds = None
    truths = None
    with torch.no_grad():
        for didx, data in enumerate(_tdl):
            ipt, ground_truth = data
            pred = _model(ipt.double())

            if didx == 0:
                preds = pred.cpu().numpy()
                truths = ground_truth.cpu().numpy()
            else:
                preds = np.hstack((preds, pred.cpu().numpy()))
                truths = np.hstack((truths, ground_truth.cpu().numpy()))

    mse = SKM.mean_squared_error(truths, preds)
    print(f'mse: {mse}')
    if to_nep == True:
        _nep['test/mse'] = mse



def test_classification(_model, _testdata, batch_size = 16, _nep=None):
    _tdl = torch.utils.data.DataLoader(_testdata, batch_size = batch_size, shuffle=True, generator=torch.Generator(device=device))
    to_nep = _nep != None
    _model.eval()
    tot_loss = 0.
    preds = None
    truths = None
    with torch.no_grad():
        for didx, data in enumerate(_tdl):
            ipt, ground_truth = data
            pred = _model(ipt)

            cur_preds = torch.argmax(pred,axis=1).cpu().numpy()
            cur_truths = ground_truth.cpu().numpy()
            if didx == 0:
                preds = cur_preds
                truths = cur_truths
            else:
                preds = np.hstack((preds, cur_preds))
                truths = np.hstack((truths, cur_truths))
    _acc = SKM.accuracy_score(truths, preds)
    _f1macro = SKM.f1_score(truths, preds, average='macro')
    _f1micro = SKM.f1_score(truths, preds, average='micro')
    class_truths = [rev_classdict[x] for x in truths]
    class_preds = [rev_classdict[x] for x in preds]
    print(f'accuracy: {_acc}, f1macro: {_f1macro}, f1micro: {_f1micro}')
    _cm = SKM.confusion_matrix(class_truths, class_preds)
    _cmd = None
    if params['split'] != 1:
        all_labels = set(class_truths).union(set(class_preds))
        class_arr2 = [x for x in class_arr if x in all_labels]
        _cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=class_arr2)
    else:
        _cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=class_arr)
    _cmd.plot()
    timestamp = int(time.time()*1000)
    if os.path.exists(res_dir) == False:
        os.makedirs(res_dir)
    cm_fname = f'{timestamp}-cm.png' 
    cm_path = os.path.join(res_dir, cm_fname)
    plt.savefig(cm_path)
    if to_nep == True:
        _nep['test/acc'] = _acc
        _nep['test/f1macro'] = _f1macro
        _nep['test/f1micro'] = _f1micro
        _nep[f'test/cm-{timestamp}'].upload(cm_path)
    plt.clf()
    #print(preds)
    #print(truths)


#nep = TN.init(params)
nep = None
if to_nep == True:
    nep = TN.init(params)
train_valid(model, train_data, valid_data, loss_fn, optim, batch_size=bs, epochs=num_epochs, _nep=nep)

if classification == True:
    test_classification(model, test_data, batch_size = bs, _nep = nep)
else:
    test_regression(model, test_data, batch_size = bs, _nep = nep)
if nep != None:
    nep.stop()
