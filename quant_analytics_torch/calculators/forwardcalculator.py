# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.models import models
from quant_analytics_torch.instruments import currencies
import datetime
import torch

class BaseCalculator(torch.nn.Module):
    def __init__(self, instrument : instruments.InstrumentBase, model : models.Model):
        super().__init__()
        self.instrument = instrument
        self.model = model

    def calculate(self):
        return None

class ForwardCalculator(BaseCalculator):
    def __init__(self, instrument : instruments.Forward, model : models.Model):
        super().__init__(instrument, model)

    def calculate(self):
        t = self.model.dateToTime(self.instrument.maturity)
        df = self.model.discountfactors[self.instrument.ccy.toString()].discountFactor(t)
        fwd =self.model.forwards[self.instrument.inst.name].forward(t)
        return df*(fwd-self.instrument.strike)


if __name__ == '__main__':
    

    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)
    inst = instruments.Asset("SPX", currencies.USD)
    fwd = instruments.Forward("SPX-1", inst, maturity=datetime.datetime(2023,6,12), strike=100., ccy=currencies.USD )

    model = models.fillSampleModel(inst)

    cal = ForwardCalculator(fwd, model)

    v = cal.calculate()

    v.backward(create_graph=True)

    for name, param in model.named_parameters():
        print(name)
        print(param.getName())
        print(param.getValue())
        print(param.getValue().grad)
        param.grad = None
