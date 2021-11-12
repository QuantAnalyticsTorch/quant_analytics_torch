from quant_analytics_torch.analytics import constants
import torch

def hyperbolic(x : torch.tensor) -> torch.tensor:
    return (x + torch.sqrt(1. + x*x))/2.

def hyperbolic_prime(x : torch.tensor) -> torch.tensor:
    return (1 + x/torch.sqrt(1. + x*x))/2.

def soft_max_hyperbolic(x,eps=constants.EPSILON):
    return hyperbolic(x/eps)*eps

def soft_heavy_side_hyperbolic(x,eps=constants.EPSILON):
    return hyperbolic_prime(x/eps)