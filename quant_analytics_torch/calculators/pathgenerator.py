import torch

torch.manual_seed(3)

class PathGenerator(torch.nn.Module):
    def __init__(self, seq_len=2):
        self.seq_len = seq_len
        self.input_dim = 1
        self.sigma = 0.2

    def paths(self, batch_size):

        # Initial model price is 0
        x0 = torch.zeros( batch_size,  self.seq_len + 1,  self.input_dim)

        # One step simulation
        dx = torch.randn(  batch_size,  self.seq_len,  self.input_dim)

        for k in range(self.seq_len):
            x0[:,k+1,:] = x0[:,k,:] + self.sigma * dx[:,k,:]

        return x0
