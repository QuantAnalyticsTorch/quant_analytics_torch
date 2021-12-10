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
        self.volatilities = torch.nn.ModuleDict({})        

    def dateToTime(self, date):
        return (date - self.modelDate).days / 365.

    def datesToTimes(self, dates):
        times = [(it - self.modelDate).days / 365. for it in dates]
        return times