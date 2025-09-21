# https://github.com/p-lambda/jukemir/blob/241555645682d18fb909775c54014edfc3a03163/jukemir/probe/__init__.py

# https://github.com/brown-palm/syntheory/blob/main/probe/probes.py
# --- essentially, dropout -> n x (linear -> relu -> dropout) -> linear

#https://github.com/brown-palm/syntheory/blob/4f222359e750ec55425c12809c1a0358b74fce49/probe/main.py#L145  
# --- use one output for regression

# https://github.com/brown-palm/syntheory/blob/4f222359e750ec55425c12809c1a0358b74fce49/tests/probe/test_probes.py#L30
# used layer 38 for regression

from collections import OrderedDict
from torch import nn
import os
import util as UM
import torch

class LinearProbe(nn.Module):
    def __init__(self, in_dim=4800, hidden_layers = [512],num_classes=10, dropout = 0.5, initial_dropout = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        d_idx = 0
        cur_layers = []
        if initial_dropout == True:
            cur_layers.append(( f'dropout_{d_idx}', nn.Dropout(p=dropout)))
            d_idx += 1
        prev_dim = in_dim
        h_idx = 0
        r_idx = 0
        for hdim in hidden_layers:
            cur_layers.append((f'linear_{h_idx}',nn.Linear(prev_dim, hdim)))
            cur_layers.append( (f'relu_{r_idx}', nn.ReLU()))
            cur_layers.append( (f'dropout_{d_idx}', nn.Dropout(p=dropout)))
            prev_dim = hdim
            h_idx += 1
            r_idx += 1
            d_idx += 1
        cur_layers.append( ( f'linear_{h_idx}', nn.Linear(prev_dim, num_classes)))

        self.layers = nn.Sequential(OrderedDict(cur_layers))
    def forward(self, x):
        out = self.layers(x)
        return out
