# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.interpolators import interpolate
from quant_analytics_torch.analytics import constants

import torch

def test_interpolator():
    a = torch.tensor([0.,1.,2.],requires_grad=True, dtype=torch.double)
    b = torch.tensor([0.,1.,4.],requires_grad=True, dtype=torch.double)
    x = torch.tensor(1.5,requires_grad=True, dtype=torch.double)

    ip = interpolate.LinearInterpolator(a,b)

    v = ip(1.5)

    assert abs(v - 2.5) < constants.EPSILON

    dx, = torch.autograd.grad(v, b, create_graph=True, retain_graph=True, allow_unused=True)

    assert abs(dx[1] - 0.5) < constants.EPSILON


if __name__ == '__main__':
    test_interpolator()