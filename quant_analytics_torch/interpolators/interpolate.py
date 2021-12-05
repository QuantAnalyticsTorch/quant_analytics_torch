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