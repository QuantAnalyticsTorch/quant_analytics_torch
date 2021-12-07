# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.instruments import currencies

import datetime


class InstrumentBase(object):
    def __init__(self):
        super().__init__()

    def type(self):
        return self.__class__.__name__

    def id(self):
        return None

    def ccy(self):
        return None

class CashDeposit(InstrumentBase):
    def __init__(self, name : str, ccy : currencies, maturity : datetime.datetime):
        super().__init__()        
        self.name = name
        self.ccy = ccy
        self.maturity = maturity

    def id(self):
        return { self.type() : { self.ccy.toString() : self.maturity } }

    def ccy(self):
        return self.ccy

class Asset(InstrumentBase):
    def __init__(self, name : str, ccy : currencies):
        super().__init__()        
        self.name = name
        self.ccy = ccy

    def id(self):
        return { self.type() : { self.name : self.ccy.toString() } }

    def ccy(self):
        return self.ccy

class Forward(InstrumentBase):
    def __init__(self, name : str, inst : InstrumentBase, maturity = datetime.datetime.now(), strike = float('NaN'), ccy = currencies.Currency() ):
        super().__init__()        
        self.name = name
        self.inst = inst
        self.maturity = maturity
        self.strike = strike
        self.ccy = ccy

    def id(self):
        return { self.type() : { self.inst.name : { self.maturity : self.strike } } }

    def ccy(self):
        return self.ccy


class EuropeanOption(InstrumentBase):
    def __init__(self, name : str, inst : InstrumentBase, maturity : str, strike : float, ccy = currencies.Currency()):
        super().__init__()        
        self.name = name
        self.inst = inst
        self.maturity = maturity
        self.strike = strike
        self.ccy = ccy        

    def id(self):
        return { self.type() : { self.inst.name : { self.maturity : self.strike } } }

    def ccy(self):
        return self.ccy


if __name__ == '__main__':
    inst = Asset("SPX", currencies.USD)
    fwd = Forward("SPX-2011", inst, datetime.datetime(2021,12,12),None)
    option = EuropeanOption("SPX-X-Y", inst, '2021-11-11', 100)
    print(inst.type())
    print(inst.id())    

    print(fwd.type())
    print(fwd.id())    

    print(option.type())
    print(option.id())    

    cashDepo = CashDeposit("USD-1y", currencies.USD, datetime.datetime(2022,12,12))

    print(cashDepo.type())
    print(cashDepo.id())    
