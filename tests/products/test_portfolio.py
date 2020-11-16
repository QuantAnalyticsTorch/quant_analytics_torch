import torch

from quant_analytics_torch.products.europeancalloption import EuropeanCallOption
from quant_analytics_torch.products.deltahedge import DeltaHedge
from quant_analytics_torch.products.portfolio import Portfolio

torch.set_printoptions(precision=16)

EPSILON = 0.00000001

def test_european_option():
    torch.manual_seed(3)

    europeancalloption = EuropeanCallOption(1, 0)

    stateVector = torch.randn(size=(100,1,1))
    decisionVector = None

    payoff = europeancalloption.payoff(stateVector, decisionVector)

    result = torch.mean(payoff)

    assert (result - 0.4015288054943085) < EPSILON

def test_delta_hedge():
    torch.manual_seed(3)

    deltahedge = DeltaHedge(2, 1)

    states = deltahedge.states()

    stateVector = torch.randn(size=(100,states.size()[0],1))
    decisionVector = torch.randn(size=(100,states.size()[0],1))    

    payoff = deltahedge.payoff(stateVector, decisionVector)

    result = torch.mean(payoff)

    assert (result - (-0.1844218373298645)) < EPSILON

if __name__ == '__main__':
    test_delta_hedge()
    

