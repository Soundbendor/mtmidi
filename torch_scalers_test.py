import torch
from torch_scalers import StandardScaler as TST
from sklearn.preprocessing import StandardScaler as KST
import numpy as np
import util as UM
import os,csv

res_path = UM.by_projpath(subpath='misc_results', make_dir = True)

def partial_fit_data(_sk_scl, _t_scl, _X):

    torch_X = torch.from_numpy(_X)

    _sk_scl.partial_fit(_X)
    _t_scl.partial_fit(torch_X)



def fit_data(_sk_scl, _t_scl, _X):

    torch_X = torch.from_numpy(_X)

    _sk_scl.fit(_X)
    _t_scl.fit(torch_X)



def check_param_closeness(_sk_scl, _t_scl):
    
    X_skparam = np.array([_sk_scl.mean_, _sk_scl.var_, _sk_scl.scale_])
    X_tparam = np.array([_t_scl.mean.cpu().numpy(), _t_scl.var.cpu().numpy(), _t_scl.scale.cpu().numpy()])

    res = {}
    params_close = True
    for i,Xparam in enumerate(['mean', 'var', 'scale']):
        cur_sk = X_skparam[i]
        cur_t = X_tparam[i]
        cur_ic = np.isclose(cur_sk, cur_t)

        sk_avg = np.mean(cur_sk)
        t_avg = np.mean(cur_t)
        all_ic = np.all(cur_ic)
        res[Xparam] = {'sklearn_avg': sk_avg, 'torch_avg': t_avg, 'close': all_ic}
        if all_ic == False:
            params_close = False
        #print(f'{Xparam} avg | scikit: {sk_avg}, torch: {t_avg}, close: {all_ic}')
    return res, params_close

        
def check_transform_closeness(_sk_scl, _t_scl, _X):
    torch_X = torch.from_numpy(_X)

    tX_sk = _sk_scl.transform(_X)
    tX_t = _t_scl.transform(torch_X).cpu().numpy()

    cur_ic = np.allclose(tX_sk, tX_t)
    return cur_ic
    #print(f'txed | close: {cur_ic}')

seed = 5
batch_size = 64
dataset_size = 45000
rng = np.random.default_rng(seed=seed)
num_batches = int(float(dataset_size)/batch_size * 100)
num_feat = 4096

device = 'cpu'

if torch.cuda.is_available() == True:
    device = 'cuda'
sk_scl = KST()
t_scl = TST(device=device)


for dist in ['uniform', 'normal', 'zipf', 'wald', 'poisson', 'power', 'chisquare']:
    fname1 = f'scalers_params-{dist}-{batch_size}_{num_batches}-{dataset_size}-{device}.csv'
    fname2 = f'scalers_txed-{dist}-{batch_size}_{num_batches}-{dataset_size}-{device}.csv'
    
    f1 = open(os.path.join(res_path, fname1), 'w')
    f2 = open(os.path.join(res_path, fname2), 'w')
    csvf1 = csv.writer(f1, delimiter = ',')
    csvf2 = csv.writer(f2, delimiter = ',')

    csvf1.writerow(['batch','param','sklearn_avg', 'torch_avg', 'close'])
    csvf2.writerow(['batch','close'])
    print(f'Testing with num_batches = {num_batches} and num_feat = {num_feat} on {dist}')
    for i in range(num_batches):
        batch = None
        if dist == 'uniform':
            batch = rng.uniform(-5000., 750., size=(batch_size,num_feat)).astype(np.float32)
        elif dist == 'normal':
            batch = rng.normal(loc=5.0, scale=20., size=(batch_size,num_feat)).astype(np.float32)
        elif dist == 'zipf':
            batch = rng.zipf(5.0, size=(batch_size,num_feat)).astype(np.float32)
        elif dist == 'wald':
            batch = rng.wald(5.0, 0.1, size=(batch_size,num_feat)).astype(np.float32)
        elif dist == 'poisson':
            batch = rng.poisson(30, size=(batch_size,num_feat)).astype(np.float32)
        elif dist == 'power':
            batch = rng.power(50., size=(batch_size,num_feat)).astype(np.float32)
        elif dist == 'chisquare':
            batch = rng.chisquare(5, size=(batch_size,num_feat)).astype(np.float32)
        size = batch.shape[0]

        print(f'------ Checking batch {i+1} (size: {size}) ------')
        partial_fit_data(sk_scl, t_scl, batch)
        #fit_data(sk_scl, t_scl, batch)
        pc_res, param_close = check_param_closeness(sk_scl, t_scl)
        for xparam in ['mean','var','scale']:
            cur_row1 = [i, xparam]
            for xsubparam in ['sklearn_avg','torch_avg', 'close']:
                cur_val = pc_res[xparam][xsubparam]
                if xsubparam == 'close':
                    cur_val = 1 if cur_val == True else 0
                cur_row1.append(cur_val)
                cur_row1.append(cur_val)
            csvf1.writerow(cur_row1)

        tx_close = check_transform_closeness(sk_scl, t_scl, batch)
        tx_write = 1 if tx_close == True else 0
        csvf2.writerow([i, tx_write])
        if param_close == False or tx_close == False:
            print(f'failed at batch {i}')
    f1.close()
    f2.close()
