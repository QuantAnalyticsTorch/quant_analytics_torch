# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.interpolators import interpolate
from quant_analytics_torch.models import models, discountmodels, forwardmodels, volatilitymodels
from quant_analytics_torch.analytics import constants

import torch
import datetime


def fillSampleModel(inst : instruments.Asset):
    # Fill with sample market data first
    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)

    model = models.Model(datetime.datetime.now())
    dmc = discountmodels.DiscountModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst.ccy )
    model.discountfactors[inst.ccy.toString()] = dmc
    fmc = forwardmodels.ForwardModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst)
    model.forwards[inst.name] = fmc
    vmc = volatilitymodels.SSVIVolatilityModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst)
    model.volatilities[inst.name] = vmc

    return model

def test_models():

    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)
    inst = instruments.Asset("SPX", currencies.USD)
    fwd = instruments.Forward("SPX-1", inst )

    model = fillSampleModel(inst)

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