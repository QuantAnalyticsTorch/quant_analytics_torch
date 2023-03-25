# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.analytics import blackanalytics
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.models import models, modelutil
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.calculators import basecalculator
import datetime
import torch

class EuropeanOptionCalculator(basecalculator.BaseCalculator):
    def __init__(self, instrument : instruments.EuropeanOption, model : models.Model):
        super().__init__(instrument, model)

    def calculate(self):
        t = self.model.dateToTime(self.instrument.getMaturity())
        df = self.model.discountfactors[self.instrument.getCcy().toString()].discountFactor(t)
        fwd =self.model.forwards[self.instrument.inst.name].forward(t)
        strike = self.instrument.strike
        vol_comp = self.model.volatilities[self.instrument.inst.name]
        vol = vol_comp.volatility(t,strike)
        return df*blackanalytics.black_torch(fwd, strike, torch.tensor(t), vol)
    
def test_european_option_calculator():
    
    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)
    inst = instruments.Asset("SPX", currencies.USD)
    euo = instruments.EuropeanOption("SPX-1", inst, maturity=datetime.datetime(2023,6,12), strike=100., currency=currencies.USD )

    model = modelutil.fillSampleModel(inst)

    cal = EuropeanOptionCalculator(euo, model)

    v = cal.calculate()

    print(v)

    v.backward()


if __name__ == '__main__':
    test_european_option_calculator()    
