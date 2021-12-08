# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.interpolators import interpolate

import torch
import datetime

class Model(torch.nn.Module):
    def __init__(self, modelDate : datetime.datetime):
        super(Model, self).__init__()
        self.modelDate = modelDate
        self.internalCurves = {}
        self.discountfactors = torch.nn.ModuleDict({})
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

class DiscountModelComponent(ModelComponentBase):
    def __init__(self, model : Model, marketData : marketdatarepository.MarketDataRepository, ccy : instruments.currencies):
        super().__init__()
        self.param = torch.nn.ParameterList()

        # Build the forward
        cashDepo = instruments.CashDeposit(None, ccy, datetime.datetime.now() )

        # Create an index into the market data repository
        idx = { cashDepo.type() : ccy.toString() }
        # Get the market data out of the market data repository
        md = marketData[idx]
        times = torch.zeros(size=(len(md),))
        cashdepos = torch.zeros(size=(len(md),))
        for i,it in enumerate(md):
            times[i] = model.dateToTime(it)
            cashdepos[i] = md[it].md.getValue()
            # Any market data that has been used needs to be registered
            self.param.append(md[it].md)

        self.ip = interpolate.LinearInterpolator(times,cashdepos)

    def discountFactor(self, time):
        return self.ip(time)


class ForwardModelComponent(ModelComponentBase):
    def __init__(self, model : Model, marketData : marketdatarepository.MarketDataRepository, inst : instruments.InstrumentBase):
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


def fillSampleModel(inst : instruments.Asset):
    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)

    model = Model(datetime.datetime.now())
    dmc = DiscountModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst.ccy )
    model.discountfactors[inst.ccy.toString()] = dmc
    fmc = ForwardModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst)
    model.forwards[inst.name] = fmc

    return model


if __name__ == '__main__':


    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)
    inst = instruments.Asset("SPX", currencies.USD)
    fwd = instruments.Forward("SPX-1", inst )

    model = fillSampleModel(inst)

    #model = Model(datetime.datetime.now())
    #dmc = DiscountModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst.ccy )
    #model.discountfactors[inst.ccy.toString()] = dmc
    #fmc = ForwardModelComponent(model, marketdatarepository.marketDataRepositorySingleton, inst)
    #model.forwards[inst.name] = fmc

    dmc = model.discountfactors[inst.ccy.toString()]
    fmc = model.forwards[inst.name]

    v = fmc.forward(1.5)

    # Run the graph backwards
    v.backward(create_graph=True)

    for name, param in model.named_parameters():
        print(name)
        print(param.getName())
        print(param.getValue())
        print(param.getValue().grad)
        param.grad = None

    print(model.discountfactors[inst.ccy.toString()].discountFactor(1.4))
