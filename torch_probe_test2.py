import torch
from torch_probe_model import LinearProbe
import utils_probing as UP

in_dim = 4800
num_classes = 20
x = LinearProbe(in_dim=4800, hidden_layers = [512],num_classes=10, dropout = 0.5, initial_dropout = True)

y = LinearProbe(in_dim=4800, hidden_layers = [512],num_classes=10, dropout = 0.5, initial_dropout = True)

for k in x.state_dict().keys():
    print(k, torch.allclose(x.state_dict()[k], y.state_dict()[k]))


UP.save_probe(x, model_shorthand = 'test', dataset = 'test', prefix=5, trial_number = 1)

UP.load_probe(y, model_shorthand = 'test', dataset = 'test', prefix=5, trial_number = 1)

for k in x.state_dict().keys():
    print(k, torch.allclose(x.state_dict()[k], y.state_dict()[k]))


