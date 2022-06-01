# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.interpolators import interpolate
from quant_analytics_torch.models import models, modelutil
from quant_analytics_torch.analytics import constants
from quant_analytics_torch.calculators import forwardcalculator, europeanoptioncalculator

import datetime

import torch

torch.set_printoptions(precision=16)

def test_forward_calculator():
    
    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)
    inst = instruments.Asset("SPX", currencies.USD)
    fwd = instruments.Forward("SPX-1", inst, maturity=datetime.datetime(2023,6,12), strike=100., ccy=currencies.USD )

    model = modelutil.fillSampleModel(inst)

    cal = forwardcalculator.ForwardCalculator(fwd, model)

    v = cal.calculate()
    print(v)

    assert abs(v - 2.4361986113717453) < constants.EPSILON

    v.backward(create_graph=True)

#    for name, param in model.named_parameters():
#        print(name)
#        print(param.getName())
#        print(param.getValue())
#        print(param.getValue().grad)
#        param.grad = None

def test_european_option_calculator():
    
    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)
    inst = instruments.Asset("SPX", currencies.USD)
    euo = instruments.EuropeanOption("SPX-1", inst, maturity=datetime.datetime(2023,6,12), strike=100., ccy=currencies.USD )

    model = modelutil.fillSampleModel(inst)

    cal = europeanoptioncalculator.EuropeanOptionCalculator(euo, model)
    #cal = eu.ForwardCalculator(fwd, model)

    v = cal.calculate()

    print(v)

    assert abs(v - 10.2051623952553658) < constants.EPSILON

    v.backward(create_graph=True)


#    for name, param in model.named_parameters():
#        print(name)
#        print(param.getName())
#        print(param.getValue())
#        print(param.getValue().grad)
#        param.grad = None



if __name__ == '__main__':
    test_forward_calculator()
    test_european_option_calculator()    