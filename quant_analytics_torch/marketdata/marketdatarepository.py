from quant_analytics_torch.marketdata import marketdata
from quant_analytics_torch.instruments import instruments

class MarketDataRepository(object):
    def __init__(self):
        super(MarketDataRepository, self).__init__()
        self.data = {}

    def getMarketData(self, inst : instruments.InstrumentBase):
        return self.data[inst.type()][inst.id()]

    def clear(self):
        self.data = {}

    def storeMarketData(self, marketdata : marketdata.MarketData):
        if not marketdata.inst.type() in self.data:
            self.data[marketdata.inst.type()] = {}
            self.data[marketdata.inst.type()][marketdata.inst.id()] = marketdata
        else:
            self.data[marketdata.inst.type()][marketdata.inst.type().id()] = marketdata

marketDataRepositorySingleton = MarketDataRepository()

if __name__ == '__main__':
    inst = instruments.EquitySpot("SPX")
    md = marketdata.MarketData(inst, 100.)
    marketDataRepositorySingleton.storeMarketData(md)

    md = marketDataRepositorySingleton.getMarketData(inst)
    param = md.md
    print(param.getName())
    print(param.getValue())    
