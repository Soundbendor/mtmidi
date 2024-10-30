# https://github.com/p-lambda/jukemir/blob/241555645682d18fb909775c54014edfc3a03163/jukemir/probe/__init__.py
from torch import nn

class LinearProbe(nn.Module):
    def __init__(self, in_dim=4800, hidden_layers = [512],num_classes=10, dropout = 0.5):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.layers = nn.Sequential(
                nn.Dropout(p=dropout)
                )
        prev_dim = in_dim
        for hdim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hdim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
            prev_dim = hdim
        self.layers.append(nn.Linear(prev_dim, num_classes))


    def forward(self, x):
        out = self.layers(x)
        return out
