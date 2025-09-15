import torch

class StandardScaler():
    def __init__(self,with_mean = True, with_std = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean = 0.
        self.var = 1.
        self.num_samples = 0
    def reset(self):
        self.mean = 0.
        self.var = 1.
        self.num_samples = 0


    def partial_fit(self):
        pass

    def fit(self, X):
        self.reset()
        self.partial_fit(X)


