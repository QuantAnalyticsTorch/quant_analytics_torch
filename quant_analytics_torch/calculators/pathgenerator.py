import torch

torch.manual_seed(3)

class PathGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def paths(self, batch_size):
        return None