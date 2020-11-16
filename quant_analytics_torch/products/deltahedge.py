import torch

from quant_analytics_torch.products.baseproduct import BaseProduct

class DeltaHedge(BaseProduct):
    def __init__(self, seq_len=2, maturity=1):
        super().__init__()
        self.maturity = maturity
        self.seq_len = seq_len

    def states(self):
        return torch.linspace(0, self.maturity, self.seq_len+1)
    
    def payoff(self, stateVector, decisionVector):
        batch_size = len(stateVector[:,0,0])
        pnl = torch.zeros(batch_size, 1)
        delta_zero = torch.zeros(batch_size, 1)

        for k in range(self.seq_len):
            pnl = pnl + (decisionVector[:,k,:]-delta_zero)  * stateVector[:,k,:]
            delta_zero = decisionVector[:,k,:]

        return pnl - decisionVector[:,-1,:]*stateVector[:,-1,:]
