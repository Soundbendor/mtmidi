import torch
from torch_scalers import StandardScaler as TST
from sklearn.preprocessing import StandardScaler as KST
import numpy as np

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
    X_tparam = np.array([_t_scl.mean.numpy(), _t_scl.var.numpy(), _t_scl.scale.numpy()])

    for i,Xparam in enumerate(['mean', 'var', 'scale']):
        cur_sk = X_skparam[i]
        cur_t = X_tparam[i]
        cur_ic = np.isclose(cur_sk, cur_t)

        sk_avg = np.mean(cur_sk)
        t_avg = np.mean(cur_t)
        all_ic = np.all(cur_ic)
        print(f'{Xparam} avg | scikit: {sk_avg}, torch: {t_avg}, close: {all_ic}')

        
def check_transform_closeness(_sk_scl, _t_scl, _X):
    torch_X = torch.from_numpy(_X)

    tX_sk = _sk_scl.transform(_X)
    tX_t = _t_scl.transform(torch_X).numpy()

    cur_ic = np.allclose(tX_sk, tX_t)
    print(f'txed | close: {cur_ic}')

seed = 5
rng = np.random.default_rng(seed=seed)
num_batches = 50
num_feat = 4096


sk_scl = KST()
t_scl = TST()

print(f'Testing with num_batches = {num_batches} and num_feat = {num_feat}')
for i in range(num_batches):
    cur_size = 50*(i+1)
    batch = rng.uniform(-5., 5., size=(cur_size,num_feat)).astype(np.float32)
    size = batch.shape[0]

    print(f'------ Checking batch {i+1} (size: {size}) ------')
    partial_fit_data(sk_scl, t_scl, batch)
    #fit_data(sk_scl, t_scl, batch)
    check_param_closeness(sk_scl, t_scl)
    check_transform_closeness(sk_scl, t_scl, batch)
