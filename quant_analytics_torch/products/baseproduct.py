import torch

class BaseProduct(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def states(self):
        return []

    def payoff(self, stateVector):
        return []
