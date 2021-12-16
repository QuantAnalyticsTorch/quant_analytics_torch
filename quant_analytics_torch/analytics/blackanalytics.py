import torch
import numpy as np
from scipy.stats import norm

def black(s, k, dt, v, r=0.0):
    std = v * np.sqrt(dt);
    d1 = np.log(s / k) / std + 0.5 * std;
    return s * norm.cdf(d1) - k * norm.cdf(d1 - std);

def black_vega(s, k, dt, v, r=0.0):
    std = v * np.sqrt(dt);
    d1 = np.log(s / k) / std + 0.5 * std;
    return s * norm.pdf(d1) * np.sqrt(dt);

def black_gamma(s, k, dt, v, r=0.0):
    std = v * np.sqrt(dt);
    d1 = np.log(s / k) / std + 0.5 * std;
    return norm.pdf(d1)/s/std;

def black_torch(f : torch.tensor, k : torch.tensor, dt : torch.tensor, v : torch.tensor) -> torch.tensor:
    """ Black Scholes formula """
    n = torch.distributions.Normal(0, 1).cdf
    sdt = v * torch.sqrt(dt)
    d1 = torch.log(f / k) / sdt + v * v * sdt / 2
    d2 = d1 - sdt
    return f * n(d1) - k * n(d2)

def black_torch_delta(s, k, dt, v, r):
    n = torch.distributions.Normal(0, 1).cdf
    sdt = v * torch.sqrt(dt)
    d1 = (torch.log(s / k) + (r + v * v / 2) * dt) / sdt
    return s * n(d1)

def black_torch_delta_diff(s, k, dt, v, r):
    x = black_torch(s, k, dt, v, r)
    dx, = torch.autograd.grad(x, [s], create_graph=True, retain_graph=True)
    return dx

def black_torch_vega_diff(s, k, dt, v, r):
    x = black_torch(s, k, dt, v, r)
    dx, = torch.autograd.grad(x, [v], create_graph=True, retain_graph=True)
    return dx

def impliedvolatility(p, s, k, dt, feps = 1e-8, veps = 1e-8, max_iter = 10):
    v = 0.2
    vp = 0
    pt = black(s, k, dt, v)
    it = 0

    while((np.abs(p-pt)>feps) and (np.abs(v-vp)>veps) and it < max_iter):
        vp = v
        vega = black_vega(s, k, dt, v)
        v = v - (pt-p)/vega;
        pt = black(s, k, dt, v)
        it = it + 1

    return v


if __name__ == '__main__':
    forward = 100.0;
    strike = 100.0;
    vol = 0.2;
    time = 1.0;
    r = 0.0;


    forward_t = torch.tensor([forward], requires_grad=True)
    strike_t = torch.tensor(strike, requires_grad=True)
    vol_t = torch.tensor([vol], requires_grad=True)
    time_t = torch.tensor(time, requires_grad=False)
    r_t = torch.tensor(r, requires_grad=False)

    bs = black_torch(forward_t, strike_t, time_t, vol_t, r_t)
    bsd = black_torch_delta(forward_t, strike_t, time_t, vol_t, r_t)
    bsdt = black_torch_delta_diff(forward_t, strike_t, time_t, vol_t, r_t)

    bsvt = black_torch_vega_diff(forward_t, strike_t, time_t, vol_t, r_t)
    print(bs)
    print(bsd)
    print(bsdt)    
    print(bsvt)

    bsn = black(forward, strike, time, vol)
    bsv = black_vega(forward, strike, time, vol)
    bsg = black_gamma(forward, strike, time, vol)

    print(bsn)
    print(bsv)
    print(bsg)    

    iv = impliedvolatility(bsn, forward, strike, time)

    print(iv)