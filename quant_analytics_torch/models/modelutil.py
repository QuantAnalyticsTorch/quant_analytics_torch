# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.models import models, discountmodels, forwardmodels, volatilitymodels, dynamicmodels
from quant_analytics_torch.analytics import constants

import torch
import datetime

def fillSampleModel(inst : instruments.Asset):
    # Fill with sample market data first
    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)

    model = models.Model(datetime.datetime.now())
    dmc = discountmodels.DiscountModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst.getCcy() )
    model.discountfactors[inst.getCcy().toString()] = dmc
    fmc = forwardmodels.ForwardModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst)
    model.forwards[inst.name] = fmc
    vmc = volatilitymodels.SSVIVolatilityModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst)
    model.volatilities[inst.name] = vmc
#    lmc = dynamicmodels.LognormalModelComponent()
#    model.dynamics[inst.name] = lmc

    return model
