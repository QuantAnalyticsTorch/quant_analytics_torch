import torch

from quant_analytics_torch.products.europeancalloption import EuropeanCallOption

torch.set_printoptions(precision=16)

EPSILON = 0.00000001

def test_european_option():
    torch.manual_seed(3)

    europeancalloption = EuropeanCallOption(1, 0)

    stateVector = torch.randn(size=(100,1,1))

    payoff = europeancalloption.payoff(stateVector)

    result = torch.mean(payoff[0])

    assert (result - 0.4015288054943085) < EPSILON


if __name__ == '__main__':
    test_european_option()
    

