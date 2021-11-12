from quant_analytics_torch.calculators.multivariatebrownianbridge import MultivariateBrownianBridge
from quant_analytics_torch.analytics.norminv import norminv
from quant_analytics_torch.analytics.constants import EPSILON

import torch

def test_multivariate_brownian_path():

    dim = 2
    paths = 3
    states = 2

    sobol_engine =  torch.quasirandom.SobolEngine(dim*states)

    x = sobol_engine.draw(1)
    x = sobol_engine.draw(paths)

    y = torch.transpose(norminv(x),0,1)
    y = torch.reshape(y, shape=(dim,states,paths))

    sigma1 = torch.tensor(0.3,requires_grad=True)
    sigma2 = torch.tensor(0.2,requires_grad=True)
    rho = torch.tensor(0.5,requires_grad=True)

    fm = torch.zeros(size=(states,states))

    fm[0][0] = sigma1*sigma1
    fm[0][1] = fm[1][0] = rho*sigma1*sigma2
    fm[1][1] = sigma2*sigma2

    fwd_cov = torch.zeros(size=(dim, states, states))

    for i in range(dim):
        fwd_cov[i] = fm

    multivariate_brownian = MultivariateBrownianBridge(fwd_cov)

    mpathi = multivariate_brownian.path(y, True)
    mpath = torch.sum(mpathi,dim=0)

    v = mpath[0][1]
    dx, = torch.autograd.grad(v, sigma1, create_graph=True, retain_graph=True, allow_unused=True)

    assert abs(0.0) < EPSILON


if __name__ == '__main__':
    
    test_multivariate_brownian_path()