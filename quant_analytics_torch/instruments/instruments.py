# Copyright (c) Quant Analytics. All rights reserved.
import datetime

class InstrumentBase(object):
    def __init__(self):
        super().__init__()

    def type(self):
        return self.__class__.__name__

    def id(self):
        return None

class Asset(InstrumentBase):
    def __init__(self, name : str):
        super().__init__()        
        self.name = name

    def id(self):
        return { self.type() : self.name }

class Forward(InstrumentBase):
    def __init__(self, name : str, inst : InstrumentBase, maturity : datetime.datetime):
        super().__init__()        
        self.name = name
        self.inst = inst
        self.maturity = maturity

    def id(self):
        return { self.type() : { self.inst.name : self.maturity } }

class EuropeanOption(InstrumentBase):
    def __init__(self, name : str, inst : InstrumentBase, maturity : str, strike : float):
        super().__init__()        
        self.name = name
        self.inst = inst
        self.maturity = maturity
        self.strike = strike

    def id(self):
        return { self.type() : { self.inst.name : { self.maturity : self.strike } } }

if __name__ == '__main__':
    inst = Asset("SPX")
    fwd = Forward("SPX-2011", inst, datetime.datetime(2021,12,12))
    option = EuropeanOption("SPX-X-Y", inst, '2021-11-11', 100)
    print(inst.type())
    print(inst.id())    

    print(fwd.type())
    print(fwd.id())    

    print(option.type())
    print(option.id())    

    