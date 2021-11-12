import tensorflow as tf
import torch
import numpy as np

SQRT_2 = np.sqrt(2.)

def norminv(x):
    return torch.erfinv(2.*(x-0.5))*SQRT_2