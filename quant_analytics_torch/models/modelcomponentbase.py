# Copyright (c) Quant Analytics. All rights reserved.
import torch

class ModelComponentBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
          
    def curves(self):
        return None
