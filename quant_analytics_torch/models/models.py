# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.interpolators import interpolate

import torch
import datetime

class Model(torch.nn.Module):
    def __init__(self, modelDate : datetime.datetime):
        super(Model, self).__init__()
        self.modelDate = modelDate
        self.internalCurves = {}
        self.forwards = torch.nn.ModuleDict({})

    def dateToTime(self, date):
        return (date - self.modelDate).days / 365.

    def datesToTimes(self, dates):
        times = [(it - self.modelDate).days / 365. for it in dates]
        return times

class ModelComponentBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
          
    def curves(self):
        return None

class ForwardModelComponent(ModelComponentBase):
    def __init__(self, model : Model, marketData : marketdatarepository.MarketDataRepository, inst : instruments.InstrumentBase):
        super().__init__()
        self.param = torch.nn.ParameterList()        

        # Build the forward
        fwd = instruments.Forward("SPX-1", inst )

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

    def curves(self):
        return self.ip

if __name__ == '__main__':
    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)
    inst = instruments.Asset("SPX")
    fwd = instruments.Forward("SPX-1", inst )

    model = Model(datetime.datetime.now())
    fmc = ForwardModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst)
    model.forwards[inst.name] = fmc

    ip = fmc.curves()

    v = ip[1.5]

    # Run the graph backwards
    v.backward(create_graph=True)

    for name, param in model.named_parameters():
        print(name)
        print(param.getName())
        print(param.getValue())
        print(param.getValue().grad)
        param.grad = None

