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
        return self.name

if __name__ == '__main__':
    es = EquitySpot("SPX")
    print(es.type())
    print(es.id())    