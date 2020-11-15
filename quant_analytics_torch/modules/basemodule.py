import torch

class BaseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_seq):
        return None