from quant_analytics_torch.calculators.univariatebrownianbridge import UnivariateBrownianBridge
from quant_analytics_torch.analytics.norminv import norminv
from quant_analytics_torch.analytics.constants import EPSILON

import torch

def test_brownian_path():
    
    dim =  2
    paths = 3
    states = 2

    brownian = UnivariateBrownianBridge(dim)

    sobol_engine =  torch.quasirandom.SobolEngine(dim*states)

    x = sobol_engine.draw(1)
    x = sobol_engine.draw(paths)

    y = torch.transpose(norminv(x),0,1)
    y = torch.reshape(y, shape=(dim,states,paths))

    path = torch.zeros(size=(dim,states,paths))

    brownian.path(path, y, True)

    assert abs(0.0) < EPSILON

if __name__ == '__main__':
    
    test_brownian_path()