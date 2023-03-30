import torch
import numpy as np

SQRT_2 = np.sqrt(2.)

def norminv(x):
    """ Inverse cumulative normal function`
    
    .. _inverse_cumulative_normal:
    
    """
    return torch.erfinv(2.*(x-0.5))*SQRT_2

def cumnorm(x):
    """ Cumulative normal function`
    
    .. _cumulative_normal:
    
    """
    return (1.0 + torch.erf(x / SQRT_2)) / 2.0