class InstrumentBase(object):
    def __init__(self):
        super().__init__()

    def type(self):
        return self.__class__.__name__

    def id(self):
        return None

class EquitySpot(InstrumentBase):
    def __init__(self, name : str):
        super().__init__()        
        self.name = name

    def id(self):
        return { self.type() : self.name }

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
    inst = EquitySpot("SPX")
    option = EuropeanOption("SPX-X-Y", inst, '2021-11-11', 100)
    print(inst.type())
    print(inst.id())    

    print(option.type())
    print(option.id())    