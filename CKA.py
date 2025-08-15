import torch

class HSIC():
    def __init__(self, kernel_type = 'linear', device = 'cpu'):
        self.device = device
        self.kernel_type = kernel_type
        self.num_batches = torch.tensor(0, dtype=torch.int, device=device)
        self.accum_hsic = torch.tensor(0., dtype=torch.float64, device=device)


    def reset(self):
        self.num_batches = torch.tensor(0, dtype=torch.int, device=self.device)
        self.accum_hsic = torch.tensor(0., dtype=torch.float64, device=self.device)

    def calc_kernel(self, X,Y):
        ret = 0.
        if self.kernel_type == 'linear':
            ret =  X @ Y.T
        return ret

    def update(self, X,Y):
        cur_sum = 0.
        if X.shape[0] == Y.shape[0]:
            N = X.shape[0]
            diag_mask = torch.ones((N,N)) - torch.eye(N)
            K = self.calc_kernel(X,X)
            L = self.calc_kernel(Y,Y)
            K_bar = torch.mul(K, diag_mask)
            L_bar = torch.mul(L, diag_mask)
            KL_bar = K_bar @ L_bar
            tr_KL_bar = torch.trace(KL_bar)
            cur_sum = (1./(N*(N-3))) * (tr_KL_bar + ((torch.sum(K_bar) * torch.sum(L_bar))/((N-1)*(N-2))) + ((2. * torch.sum(KL_bar))/(N-2)))
            self.accum_hsic += cur_sum
            self.num_batches += 1
    def get_value(self):
        return self.accum_hsic.item()/torch.tensor([0., self.num_batches.item()]).max().item()

class CKA():
    def __init__(self, kernel_type='linear', device='cpu'):
        self.kernel_type=kernel_type
        self.device=device
        self.hsic_KL = HSIC(kernel_type=kernel_type,device=device)
        self.hsic_KK = HSIC(kernel_type=kernel_type,device=device)
        self.hsic_LL = HSIC(kernel_type=kernel_type,device=device)

    def reset(self):
        self.hsic_KL.reset()
        self.hsic_KK.reset()
        self.hsic_LL.reset()

    def update(self, X,Y):
        if X.shape[0] == Y.shape[0]:
            self.hsic_KL.update(X,Y)
            self.hsic_KK.update(X,X)
            self.hsic_LL.update(Y,Y)
    
    def get_value(self):
        return self.hsic_KL.get_value()/torch.sqrt(self.hsic_KK.get_value() * self.hsic_LL.get_value())

