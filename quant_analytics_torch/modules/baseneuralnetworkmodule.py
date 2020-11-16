import torch

class BaseNeuralNetworkModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_seq):
        return None