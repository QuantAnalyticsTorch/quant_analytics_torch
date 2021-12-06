# Copyright (c) Quant Analytics. All rights reserved.
import torch
from quant_analytics_torch.instruments import instruments

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


if __name__ == '__main__':
    inst = instruments.Asset("SPX")
    md = MarketData(inst, 100., { 'ccy' : "EUR" })
    #print(md.inst.type())
    #print(md.inst.id())
    #print(md.details)    
    param = md.md
    #print(param.getName())
    print(param.getValue())

    w = param.getValue()*param.getValue()
    print(w)
    w.backward()
    #w.gradient(create_graph=True)

    print(param.getValue().grad)

    #x = 100.
    #mp = MarketParameter(torch.tensor(x, requires_grad=True), True, { 'type' : inst.type(), 'id' : inst.id() })
    #print(mp)
    #w = mp.getValue()*mp.getValue()
    #w.backward()
    #z = mp.getValue()
    #print(z.grad)