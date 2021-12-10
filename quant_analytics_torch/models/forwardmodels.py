# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.interpolators import interpolate
from quant_analytics_torch.models import models, modelcomponentbase

import torch
import datetime

class ForwardModelComponent(modelcomponentbase.ModelComponentBase):
    def __init__(self, model : models.Model, marketData : marketdatarepository.MarketDataRepository, inst : instruments.InstrumentBase):
        super().__init__()
        self.param = torch.nn.ParameterList()        

        # Build the forward
        fwd = instruments.Forward(None, inst )

        # Create an index into the market data repository
        idx = { fwd.type() : inst.name }
        # Get the market data out of the market data repository
        md = marketData[idx]
        times = torch.zeros(size=(len(md),))
        fwds = torch.zeros(size=(len(md),))
        for i,it in enumerate(md):
            times[i] = model.dateToTime(it)
            mdi = next(iter(md[it].values()))
            fwds[i] = mdi.md.getValue()
            # Any market data that has been used needs to be registered
            self.param.append(mdi.md)

        self.ip = interpolate.LinearInterpolator(times,fwds)

    def forward(self, time):
        return self.ip(time)
