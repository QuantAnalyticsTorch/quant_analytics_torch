# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.interpolators import interpolate
from quant_analytics_torch.models import models, modelcomponentbase

import torch
import datetime

class DiscountModelComponent(modelcomponentbase.ModelComponentBase):
    def __init__(self, model : models.Model, marketData : marketdatarepository.MarketDataRepository, ccy : instruments.currencies):
        super().__init__()
        self.param = torch.nn.ParameterList()

        # Build the forward
        cashDepo = instruments.CashDeposit(None, ccy, datetime.datetime.now() )

        # Create an index into the market data repository
        idx = { cashDepo.type() : ccy.toString() }
        # Get the market data out of the market data repository
        md = marketData[idx]
        times = torch.zeros(size=(len(md),),dtype=torch.double)
        cashdepos = torch.zeros(size=(len(md),),dtype=torch.double)
        for i,it in enumerate(md):
            times[i] = model.dateToTime(it)
            cashdepos[i] = md[it].md.getValue()
            # Any market data that has been used needs to be registered
            self.param.append(md[it].md)

        self.ip = interpolate.LinearInterpolator(times,cashdepos)

    def discountFactor(self, time):
        return self.ip(time)