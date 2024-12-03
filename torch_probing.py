# https://github.com/brown-palm/syntheory/blob/main/probe/probes.py

import torch
from torch import nn
from torch_activations_dataset import STPActivationsData 
from torch_sthf_dataset import STHFTempiData
from torch_probe_model import LinearProbe
import numpy as np
import torch_nep as TN
import sklearn.metrics as SKM
import os
import matplotlib.pyplot as plt
import time
import sys
import polyrhythms as PL
import util as um
import tempi as TP
import sys
import tomllib
from distutils.util import strtobool

torch.manual_seed(5)
sett = um.read_toml(sys.argv)
shuffle = True
#poly_pair_arr = [(i,j) for i in range(2,max_num+1) for j in range(2,max_num+1) if (np.gcd(i,j) == 1 and i < j)]
#poly_pairs = { (i,j): (i/j) for i in range(2,max_num+1) for j in range(2,max_num+1) if (np.gcd(i,j) == 1 and i < j)}
#poly_tups = [((i,j),x) for (i,j),x in poly_pairs.items()]
#ptsort = sorted(poly_tups, key=itemgetter(1))
figsize = 15
thresh = sett['thresh']
data_debug = False
to_nep = True
split = sett['split']
classification = sett['cls']
bs = sett['bs']
#lr = 1e-3
#num_epochs = 500
num_epochs = sett['num_epochs']
lr = sett['lr']
#num_epochs = 500
dropout = 0.5
num_hidden = sett['num_hidden']
dataset = sett['dataset']
hidden_layers = []
if num_hidden > 0:
    hidden_layers = [512] * num_hidden

act_mk = sett['act_kind']
act_lh = um.act_longhand[act_mk]
model_type = um.model_type[act_lh]
in_dim = um.act_layer_dim[act_lh]
act_folder = um.act_folder[act_lh]
num_layers = um.model_num_layers[model_type]
layer_idx = min(num_layers - 1, sett['layer_idx'])
hidden_layer_str = "["+",".join([str(y) for y in hidden_layers]) + "]"
res_dir = os.path.join(um.script_dir, "res")
user_folder = os.path.expanduser("~" + os.getenv("USER")) 
#data_folder = os.path.join(user_folder, "ds", "jukebox_acts_36")
params = {'batch_size': bs, 'num_epochs': num_epochs, 'lr': lr, 'dropout': dropout, 'use_cuda': True,
          'thresh': thresh, 'hidden_layers': hidden_layer_str, 'in_dim': in_dim,
          'act_folder': act_folder, 'model_type': model_type, 'act_lh': act_lh,
          'dataset': dataset}
device ='cpu'

if torch.cuda.is_available() == True and params["use_cuda"] == True:
    device = 'cuda'
    torch.cuda.empty_cache()
    torch.set_default_device(device)

classdict = None
rev_classdict = None
num_classes = None
classlist_sorted = None
classset_aug = None
if dataset == 'polyrhythms':
    csvfile = os.path.join(um.script_dir, 'csv', 'polyrhy_split1.csv')
    if params['split'] != 1:
        print('running split 2')
        csvfile = os.path.join(um.script_dir, 'csv', 'polyrhy_split2.csv')
    data_folder = os.path.join(um.script_dir, "acts", act_folder)
    classdict = PL.polystr_to_idx
    num_classes = PL.num_poly
    if classification == False:
        #num_epochs = 500
        #num_epochs = 5
        #lr = 1e-3
        classdict = PL.reg_polystr_to_idx
        num_classes += 1 # to account for no label

    train_data = STPActivationsData(csvfile = csvfile, device=device, data_folder = data_folder, classdict = classdict, num_classes = num_classes, set_type='train', layer_idx = layer_idx, classification = classification)
    valid_data = STPActivationsData(csvfile = csvfile,  device=device, data_folder = data_folder, classdict = classdict, num_classes = num_classes, set_type='val', layer_idx = layer_idx, classification = classification)
    test_data = STPActivationsData(csvfile = csvfile,  device=device, data_folder = data_folder, classdict = classdict, num_classes = num_classes, set_type='test', layer_idx = layer_idx, classification = classification)

elif dataset =='tempi':
    csvfile = os.path.join(um.script_dir, 'hf_csv', 'tempi.csv')
    data_folder = os.path.join(um.script_dir, "hf_acts", act_folder)
    class_binsize = sett['class_binsize']

    classdict, rev_classdict, classlist_sorted, classset_aug = TP.get_class_medians(class_binsize)
    train_data = STHFTempiData(csvfile = csvfile, device=device, data_folder = data_folder,  set_type='train', layer_idx = layer_idx, class_binsize = class_binsize, norm_labels = True)
    valid_data = STHFTempiData(csvfile = csvfile,  device=device, data_folder = data_folder, set_type='val', layer_idx = layer_idx, class_binsize = class_binsize, norm_labels = True)
    test_data = STHFTempiData(csvfile = csvfile,  device=device, data_folder = data_folder, set_type='test', layer_idx = layer_idx, class_binsize = class_binsize, norm_labels = True)
    
    # add null class
    num_classes = len(classlist_sorted) + 1

params.update(sett)

if data_debug == True:
    num_train = len(train_data)
    num_valid = len(valid_data)
    num_test = len(test_data)
    print(f"num_train: {num_train}, num_valid: {num_valid}, num_test: {num_test}")
    sys.exit()
model = None
loss_fn = None
if classification == True:
    model = LinearProbe(in_dim = in_dim, num_classes=PL.num_poly,hidden_layers = hidden_layers, dropout = dropout).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
else:
    model = LinearProbe(in_dim = in_dim, num_classes=1,hidden_layers = hidden_layers, dropout = dropout).float().to(device)
    loss_fn = torch.nn.MSELoss(reduction='mean')
optim = torch.optim.Adam(model.parameters(), lr=lr)

def train_epoch(_model, _tdl, _lossfn, _optimfn):
    _model.train(True)
    tot_loss = 0.
    for didx, data in enumerate(_tdl):
        _optimfn.zero_grad()
        _loss = None
        if classification == True:
            ipt, ground_truth = data
            pred = _model(ipt)
            _loss = _lossfn(pred, ground_truth)
        else:
            ipt, ground_truth, ground_label = data
            pred = _model(ipt.float())
            #print('train', pred.shape, pred.dtype, ground_truth.shape, ground_truth.dtype) 
            _loss = _lossfn(pred.flatten().float(), ground_truth.flatten().float())
        _loss.backward()
        _optimfn.step()
        tot_loss += _loss.item()

    return tot_loss

def valid_epoch(_model, _vdl, _lossfn):
    _model.eval()
    tot_loss = 0.
    with torch.no_grad():
        for didx, data in enumerate(_vdl):
            if classification == True:
                ipt, ground_truth = data
                pred = _model(ipt)
                _loss = _lossfn(pred, ground_truth)
            else:
                ipt, ground_truth, ground_label = data
                pred = _model(ipt.float())
                #print('valid', pred.shape, pred.dtype, ground_truth.shape, ground_truth.dtype) 
                _loss = _lossfn(pred.flatten().float(), ground_truth.flatten().float())

            tot_loss += _loss.item()
    return tot_loss


def train_valid(_model, _traindata, _validdata, _lossfn, _optimfn, batch_size=16, epochs=10, _nep=None):
    train_losses = []
    valid_losses = []
    to_nep = _nep != None
    print("--- training ---")
    for eidx in range(epochs):
        print(f'Epoch {eidx+1}')
        train_dl = torch.utils.data.DataLoader(_traindata, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator(device=device))
        valid_dl = torch.utils.data.DataLoader(_validdata, batch_size = batch_size, shuffle=shuffle, generator=torch.Generator(device=device))
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
    _tdl = torch.utils.data.DataLoader(_testdata, batch_size = batch_size, shuffle=shuffle, generator=torch.Generator(device=device))
    to_nep = _nep != None
    _model.eval()
    tot_loss = 0.
    preds = None
    truths = None
    pred_labels = None
    truth_labels = None
    with torch.no_grad():
        for didx, data in enumerate(_tdl):
            ipt, ground_truth, ground_label = data
            pred = _model(ipt.float())

            pred_np = pred.cpu().numpy().flatten()
            cur_pred_labels = None
            cur_pred_label_idx = None
            if dataset == 'polyrhythms':
                cur_pred_labels = [PL.get_nearest_poly(x, thresh=thresh) for x in pred_np]
                cur_pred_label_idx = np.array([PL.reg_polystr_to_idx[x] for x in cur_pred_labels])
            elif dataset == 'tempi':
                cur_pred_labels = [TP.get_nearest_bpmclass(x, classlist_sorted, thresh=thresh) for x in pred_np]
                #print('test2', pred_np.shape, pred_np.dtype, len(cur_pred_labels))
                cur_pred_label_idx = np.array([classdict[x] for x in cur_pred_labels])

            if didx == 0:
                preds = pred_np
                truths = ground_truth.cpu().numpy().flatten()
                truth_labels = ground_label.cpu().numpy().flatten()
                #print('test', preds.shape, preds.dtype, truths.shape, truths.dtype) 
                pred_labels = cur_pred_label_idx
            else:
                preds = np.hstack((preds, pred_np))
                truths = np.hstack((truths, ground_truth.cpu().numpy().flatten()))
                truth_labels = np.hstack((truth_labels, ground_label.cpu().numpy().flatten()))
                pred_labels = np.hstack((pred_labels, cur_pred_label_idx))
    _mse = SKM.mean_squared_error(truths, preds)
    _r2 = SKM.r2_score(truths, preds)
    _mae = SKM.mean_absolute_error(truths, preds)
    _ev = SKM.explained_variance_score(truths, preds)
    _medae = SKM.median_absolute_error(truths, preds)
    _maxerr = SKM.max_error(truths, preds)
    _mape = SKM.mean_absolute_percentage_error(truths, preds)
    _rmse = SKM.root_mean_squared_error(truths, preds)
    _d2ae = SKM.d2_absolute_error_score(truths, preds)
    _acc = SKM.accuracy_score(truth_labels, pred_labels)
    _f1macro = SKM.f1_score(truth_labels, pred_labels, average='macro')
    _f1micro = SKM.f1_score(truth_labels, pred_labels, average='micro')

    print(f'mse: {_mse}, r2: {_r2}, mae: {_mae}, exp_var: {_ev}, median_ae: {_medae}')
    print(f'max_err: {_maxerr}, mape: {_mape}, rmse:{_rmse}, d2ae: {_d2ae}')
    print(f'accuracy: {_acc}, f1macro: {_f1macro}, f1micro: {_f1micro}')
    
    class_truths = None
    class_preds = None

    if dataset == 'polyrhythms':
        class_truths = [PL.reg_rev_polystr_to_idx[x] for x in truth_labels]
        class_preds = [PL.reg_rev_polystr_to_idx[x] for x in pred_labels]
    elif dataset == 'tempi':
        class_truths = [rev_classdict[x] for x in truth_labels]
        class_preds = [rev_classdict[x] for x in pred_labels]
    _cm = SKM.confusion_matrix(class_truths, class_preds)
    _cmd = None
    if params['split'] != 1 or dataset != 'polyrhythms':
        all_labels = set(class_truths).union(set(class_preds))
        class_arr2 = None
        if dataset == 'polyrhythms':
            class_arr2 = [x for x in PL.reg_class_arr if x in all_labels]
        else:
            class_arr2 = [x for x in classset_aug if x in all_labels]
        _cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=class_arr2)
    else:
        _cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=PL.reg_class_arr)
    _cmd.plot()
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    _cmd.plot(ax=ax)
    timestamp = int(time.time()*1000)
    if os.path.exists(res_dir) == False:
        os.makedirs(res_dir)
    cm_fname = f'{timestamp}-cm.png' 
    cm_path = os.path.join(res_dir, cm_fname)
    plt.savefig(cm_path)

    if to_nep == True:
        _nep['test/mse'] = _mse
        _nep['test/rmse'] = _rmse
        _nep['test/r2'] = _r2
        _nep['test/mae'] = _mae
        _nep['test/med_ae'] = _medae
        _nep['test/maxerr'] = _maxerr
        _nep['test/expvar'] = _ev
        _nep['test/mape'] = _mape
        _nep['test/d2ae'] = _d2ae
        _nep['test/acc'] = _acc
        _nep['test/f1macro'] = _f1macro
        _nep['test/f1micro'] = _f1micro
        _nep[f'test/cm-{timestamp}'].upload(cm_path)

    plt.clf()

def test_classification(_model, _testdata, batch_size = 16, _nep=None):
    _tdl = torch.utils.data.DataLoader(_testdata, batch_size = batch_size, shuffle=shuffle, generator=torch.Generator(device=device))
    to_nep = _nep != None
    _model.eval()
    tot_loss = 0.
    preds = None
    truths = None
    with torch.no_grad():
        for didx, data in enumerate(_tdl):
            ipt, ground_truth = data
            pred = _model(ipt)

            cur_preds = torch.argmax(pred,axis=1).cpu().numpy().flatten()
            cur_truths = ground_truth.cpu().numpy().flatten()
            if didx == 0:
                preds = cur_preds
                truths = cur_truths
            else:
                preds = np.hstack((preds, cur_preds))
                truths = np.hstack((truths, cur_truths))
    _acc = SKM.accuracy_score(truths, preds)
    _f1macro = SKM.f1_score(truths, preds, average='macro')
    _f1micro = SKM.f1_score(truths, preds, average='micro')
    class_truths = [PL.rev_polystr_to_idx[x] for x in truths]
    class_preds = [PL.rev_polystr_to_idx[x] for x in preds]
    print(f'accuracy: {_acc}, f1macro: {_f1macro}, f1micro: {_f1micro}')
    _cm = SKM.confusion_matrix(class_truths, class_preds)
    _cmd = None
    if params['split'] != 1:
        all_labels = set(class_truths).union(set(class_preds))
        class_arr2 = [x for x in PL.class_arr if x in all_labels]
        _cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=class_arr2)
    else:
        _cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=PL.class_arr)
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    _cmd.plot(ax=ax)
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
