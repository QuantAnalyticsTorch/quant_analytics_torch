# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.interpolators import interpolate
from quant_analytics_torch.models import models, modelutil
from quant_analytics_torch.analytics import constants

import torch
import datetime

def test_models():

    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)
    inst = instruments.Asset("SPX", currencies.USD)
    fwd = instruments.Forward("SPX-1", inst )

    model = modelutil.fillSampleModel(inst)

    # Get the model components
    dmc = model.discountfactors[inst.ccy.toString()]
    fmc = model.forwards[inst.name]
    vmc = model.volatilities[inst.name]    

    #v = fmc.forward(1.5)
    v = vmc.volatility(1.5, 100)   

    assert abs(v - 102.700) < constants.EPSILON

    # Run the graph backwards
    v.backward(create_graph=True)

#    for name, param in model.named_parameters():
#        print(name)
#        print(param.getName())
#        print(param.getValue())
#        print(param.getValue().grad)
#        param.grad = None

#    print(model.discountfactors[inst.ccy.toString()].discountFactor(1.4))


if __name__ == '__main__':
    test_models()