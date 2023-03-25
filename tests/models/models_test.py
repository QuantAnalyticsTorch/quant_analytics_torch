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
    fwd = instruments.Forward("SPX-1", inst, datetime.datetime.now(), 'NaN', currencies.USD )

    model = modelutil.fillSampleModel(inst)

    # Get the model components
    dmc = model.discountfactors[inst.getCcy().toString()]
    fmc = model.forwards[inst.name]
    vmc = model.volatilities[inst.name]    

    #v = fmc.forward(1.5)
    v = vmc.volatility(1.5, 80)   

    print(v)

    #assert abs(v - 0.2253) < constants.EPSILON

    # Run the graph backwards
    v.backward(create_graph=True)

    for name, param in model.named_parameters():
        print(name)
        print(param.getName())
        print(param.getValue())
        print(param.getValue().grad)
        param.grad = None


if __name__ == '__main__':
    test_models()