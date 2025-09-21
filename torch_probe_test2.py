import torch
from torch_probe_model import LinearProbe
import utils_probing as UP
import os,csv
import util as UM

res_path = UM.by_projpath(subpath='misc_results', make_dir = True)
in_dim = 4800
num_classes = 20

device = 'cpu'

if torch.cuda.is_available() == True:
    device = 'cuda'


x = LinearProbe(in_dim=4800, hidden_layers = [512],out_dim=10, dropout = 0.5, initial_dropout = True).to(device)

y = LinearProbe(in_dim=4800, hidden_layers = [512],out_dim=10, dropout = 0.5, initial_dropout = True).to(device)

fname1 = f'probe_savetest.csv'
f1 = open(os.path.join(res_path, fname1), 'w')
csvf1 = csv.writer(f1, delimiter = ',')

csvf1.writerow(['loaded', 'param', 'close'])
int_bool = lambda x: 1 if x == True else 0
for k in x.state_dict().keys():
    is_close = torch.allclose(x.state_dict()[k], y.state_dict()[k])
    csvf1.writerow([0, k, int_bool(is_close)])


UP.save_probe(x, model_shorthand = 'test', dataset = 'test', prefix=5, trial_number = 1)

UP.load_probe(y, model_shorthand = 'test', dataset = 'test', prefix=5, trial_number = 1)

for k in x.state_dict().keys():
    is_close = torch.allclose(x.state_dict()[k], y.state_dict()[k])
    csvf1.writerow([1, k, int_bool(is_close)])


f1.close()
