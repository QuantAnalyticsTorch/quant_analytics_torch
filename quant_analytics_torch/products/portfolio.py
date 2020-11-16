import torch

from quant_analytics_torch.products.baseproduct import BaseProduct

class Portfolio(BaseProduct):
    def __init__(self, products):
        super().__init__()
        self.products = products

    def states(self):
        return []
    
    def payoff(self, stateVector, decisionVector):
        return None


