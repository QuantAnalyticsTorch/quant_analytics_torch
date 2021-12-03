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
    def __init__(self, inst : instruments.InstrumentBase, value : float):
        super(MarketData, self).__init__()
        self.inst = inst
        self.md = MarketParameter(torch.nn.Parameter(torch.tensor(value)), True, { 'type' : inst.type(), 'id' : inst.id() })


if __name__ == '__main__':
    inst = instruments.EquitySpot("SPX")
    md = MarketData(inst, 100.)
    print(inst.type())
    print(inst.id())    
    param = md.md
    print(param.getName())
    print(param.getValue())    