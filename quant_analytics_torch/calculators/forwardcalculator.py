# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.models import models, modelutil
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.calculators import basecalculator
import datetime
import torch

class ForwardCalculator(basecalculator.BaseCalculator):
    def __init__(self, instrument : instruments.Forward, model : models.Model):
        super().__init__(instrument, model)

    def calculate(self):
        t = self.model.dateToTime(self.instrument.maturity)
        df = self.model.discountfactors[self.instrument.getCcy().toString()].discountFactor(t)
        fwd =self.model.forwards[self.instrument.inst.name].forward(t)
        return df*(fwd-self.instrument.strike)


def test_forward_calculator():
    
    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)
    inst = instruments.Asset("SPX", currencies.USD)
    fwd = instruments.Forward("SPX-1", inst, maturity=datetime.datetime(2023,6,12), strike=100., currency=currencies.USD )

    model = modelutil.fillSampleModel(inst)

    cal = ForwardCalculator(fwd, model)

    v = cal.calculate()
    print(v)

    v.backward()


if __name__ == '__main__':
    test_forward_calculator()