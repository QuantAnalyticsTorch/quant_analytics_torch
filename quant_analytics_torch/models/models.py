# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments

import torch
import datetime

class BaseModel(torch.nn.Module):
    def __init__(self, data, modelDate : datetime.datetime):
        super(BaseModel, self).__init__()
        self.data = data
        self.modelDate = modelDate
        self.internalCurves = {}

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y        

class ForwardModel(BaseModel):
    def __init__(self, data, modelDate : datetime.datetime):
        super(BaseModel, self).__init__()
        self.data = data
        self.internalCurves = {}


class ModelComponentBase():
    def __init__(self):
        super(ModelComponentBase, self).__init__()
        self.data = None
    
    def requiredMarketData(self):
        return None
        
    def buildCurve(self, inst : instruments.InstrumentBase):
        return None

class ForwardModelComponent(ModelComponentBase):
    def __init__(self):
        super(ForwardModelComponent, self).__init__()
        self.data = None
           
    def buildCurve(self, marketData : marketdatarepository.MarketDataRepository, inst : instruments.InstrumentBase):
        idx = { inst.type() : inst.inst.name }
        md = marketData[idx]
        # Next is to convert dates to time and then return an interpolator
        print(md)
        return None


if __name__ == '__main__':
    marketdatarepository.fillSampleDate(marketdatarepository.marketDataRepositorySingleton)
    inst = instruments.Asset("SPX")
    fwd = instruments.Forward("SPX-1", inst )

    fmc = ForwardModelComponent()

    fmc.buildCurve(marketdatarepository.marketDataRepositorySingleton, fwd)