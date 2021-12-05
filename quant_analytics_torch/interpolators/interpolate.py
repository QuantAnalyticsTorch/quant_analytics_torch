# Copyright (c) Quant Analytics. All rights reserved.
import torch

class PiecewiseConstantInterpolator(torch.nn.Module):
    def __init__(self, xs, ys):
        super(PiecewiseConstantInterpolator, self).__init__()
        self.xs = xs
        self.ys = ys

    def __getitem__(self, x):        
        idx = torch.searchsorted(self.xs,x,right=True)
        return self.ys[idx-1]


class LinearInterpolator(torch.nn.Module):
    def __init__(self, xs, ys):
        super(LinearInterpolator, self).__init__()
        self.xs = xs
        self.ys = ys
        self.slopes = (ys[1:] - ys[:-1])/(xs[1:]-xs[:-1])

    def __getitem__(self, x):        
        idx = torch.searchsorted(self.xs,x,right=True)-1
        return self.ys[idx] + self.slopes[idx] * (x - self.xs[idx])