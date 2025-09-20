import time
import torch
import numpy as np
import utils_probing as UP
import util as UM
import os,csv
from torch_scalers import StandardScaler as TST

cur_time = int(time.time()*1000)
time_str = str(cur_time)

fname = f'{time_str}_test'
device = 'cpu'

if torch.cuda.is_available() == True:
    device = 'cuda'


seed = 5
batch_size = 64
dataset_size = 8192
rng = np.random.default_rng(seed=seed)
num_batches = int(float(dataset_size)/batch_size * 100)
num_feat = 4096

res_path = UM.by_projpath(subpath='misc_results', make_dir = True)
t_scl = TST(dim=num_feat,device=device)
t_scl2 = TST(dim=num_feat,device=device)
for batch in range(num_batches):
    batch = rng.uniform(-5000., 750., size=(batch_size,num_feat)).astype(np.float32)
    t_scl.partial_fit(torch.from_numpy(batch))

cur_mean = t_scl.mean.clone().detach()
cur_var = t_scl.var.clone().detach()
cur_scale = t_scl.scale.clone().detach()
cur_ns = t_scl.num_samples.clone().detach()

UP.save_scaler(t_scl, fname)
UP.load_scaler(t_scl2, fname)

print(t_scl.mean.data)
print(t_scl2.mean.data)
print("------")
print(t_scl.var.data)
print(t_scl2.var.data)
print("------")
print(t_scl.scale.data)
print(t_scl2.scale.data)
print("------")
print(t_scl.num_samples.data)
print(t_scl2.num_samples.data)

int_bool = lambda x: 1 if x == True else 0

mean_close = torch.allclose(t_scl.mean.data, t_scl2.mean.data)
var_close = torch.allclose(t_scl.var.data, t_scl2.var.data)
scale_close = torch.allclose(t_scl.scale.data, t_scl2.scale.data)
num_samples_close = torch.allclose(t_scl.num_samples.data, t_scl2.num_samples.data)
fname1 = f'scalers_savetest-{batch_size}_{num_batches}-{dataset_size}-{device}.csv'
f1 = open(os.path.join(res_path, fname1), 'w')
csvf1 = csv.writer(f1, delimiter = ',')

csvf1.writerow(['param', 'val1', 'val2', 'close'])
csvf1.writerow(['mean', t_scl.mean.data.mean().item(), t_scl2. mean.data.mean().item(), int_bool(mean_close)])
csvf1.writerow(['var', t_scl.var.data.mean().item(), t_scl2. var.data.mean().item(), int_bool(var_close)])
csvf1.writerow(['scale', t_scl.scale.data.mean().item(), t_scl2. scale.data.mean().item(), int_bool(scale_close)])
csvf1.writerow(['num_samples', t_scl.num_samples.data.item(), t_scl2. num_samples.data.item(), int_bool(num_samples_close)])
f1.close()
