# Copyright (c) Quant Analytics. All rights reserved.
import torch

class BaseInterpolator(torch.nn.Module):
    def __init__(self):
        super(BaseInterpolator, self).__init__()

    def __getitem__(self, x):        
        return None


class PiecewiseConstantInterpolator(BaseInterpolator):
    def __init__(self, xs, ys):
        super(PiecewiseConstantInterpolator, self).__init__()
        self.xs = xs
        self.ys = ys
        self.n = len(xs)-1

    def __call__(self, x):        
        idx = torch.searchsorted(self.xs,x,right=True)
        return self.ys[idx-1]

class LinearInterpolator(BaseInterpolator):
    def __init__(self, xs, ys):
        super(LinearInterpolator, self).__init__()
        self.xs = xs
        self.ys = ys
        self.slopes = (ys[1:] - ys[:-1])/(xs[1:]-xs[:-1])
        self.n = len(xs)-1

    def __call__(self, x):
        idx = torch.searchsorted(self.xs,x,right=True)-1
        if idx >= self.n:
            return self.ys[-1]
        elif idx < 0:
            return self[0]
        else:
            return self.ys[idx] + self.slopes[idx] * (x - self.xs[idx])