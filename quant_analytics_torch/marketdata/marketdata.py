# Copyright (c) Quant Analytics. All rights reserved.
import torch
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies

class MarketParameter(torch.nn.Parameter):
    def __new__(cls, value, require_grad, parameter_name):
        return torch.nn.Parameter.__new__(cls, value, require_grad)

    def __init__(self, value, require_grad=True, parameter_name=None):
        super().__init__()
        self.value = value
        self.parameter_name = parameter_name

    def getName(self):
        return self.parameter_name

    def getValue(self):
        return self.value

class MarketData(object):
    def __init__(self, inst : instruments.InstrumentBase, value : float, details = dict() ):
        super(MarketData, self).__init__()
        self.inst = inst
        self.md = MarketParameter(torch.tensor(value, requires_grad=True), True, { 'type' : inst.type(), 'id' : inst.id() })
        self.details = details