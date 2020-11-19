import torch

from quant_analytics_torch.calculators.pathgenerator import PathGenerator

class PathGeneratorLognormal(PathGenerator):
    def __init__(self, seq_len=2, forwardvariance=0.04, fwd=100):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = 1
        self.forwardvariance = forwardvariance
        self.sigma = torch.sqrt(forwardvariance)
        self.fwd = fwd

    def paths(self, batch_size):

        # Initial model price is 0
        x0 = self.fwd * torch.ones( batch_size,  self.seq_len + 1,  self.input_dim)
        z0 = torch.zeros( batch_size,  self.seq_len + 1,  self.input_dim)

        # One step simulation
        dx = torch.randn(  batch_size,  self.seq_len,  self.input_dim)

        for k in range(self.seq_len):
            z0[:,k+1,:] = z0[:,k,:] + self.sigma * dx[:,k,:] - self.forwardvariance/2
            x0[:,k+1,:] = self.fwd * torch.exp(z0[:,k+1,:])

        return x0
