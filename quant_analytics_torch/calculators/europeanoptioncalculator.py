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
        t = self.model.dateToTime(self.instrument.maturity)
        df = self.model.discountfactors[self.instrument.ccy.toString()].discountFactor(t)
        fwd =self.model.forwards[self.instrument.inst.name].forward(t)
        strike = self.instrument.strike
        vol_comp = self.model.volatilities[self.instrument.inst.name]
        vol = vol_comp.volatility(t,strike)
        return df*blackanalytics.black_torch(fwd, strike, torch.tensor(t), vol)