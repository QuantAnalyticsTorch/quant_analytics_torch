import torch

from quant_analytics_torch.products.baseproduct import BaseProduct

class EuropeanCallOption(BaseProduct):
    def __init__(self, maturity=1, strike=0):
        super().__init__()
        self.maturity = maturity
        self.strike = strike

    def states(self):
        return []

    def payoff(self, stateVector, decisionVector):
        return torch.max(stateVector[:,-1,:],torch.tensor(0.))
