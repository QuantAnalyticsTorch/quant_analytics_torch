from quant_analytics_torch.marketdata import marketdata
from quant_analytics_torch.instruments import instruments

def recursive_insert(d,a,x):
    for it in a.keys():
        if not it in d:
            d[it] = {}
        if isinstance(a[it],dict):
            recursive_insert(d[it],a[it],x)
        else:
            d[it].update({ a[it] : x })


class MarketDataRepository(object):
    def __init__(self):
        super(MarketDataRepository, self).__init__()
        self.data = {}

    def getMarketData(self, id):
        md = self.data
        while isinstance(id, dict):
            k = next(iter(id))
            md = md[k]
            id = id[k]
        return md[id]

    def clear(self):
        self.data = {}

    def storeMarketData(self, marketdata : marketdata.MarketData):
        recursive_insert(self.data, marketdata.inst.id(), marketdata )

marketDataRepositorySingleton = MarketDataRepository()


if __name__ == '__main__':
    inst_1 = instruments.EquitySpot("SPX")
    inst_2 = instruments.EquitySpot("AAPL")

    md_1 = marketdata.MarketData(inst_1, 100.)
    marketDataRepositorySingleton.storeMarketData(md_1)

    md_2 = marketdata.MarketData(inst_2, 100.)
    marketDataRepositorySingleton.storeMarketData(md_2)


    md = marketDataRepositorySingleton.getMarketData(inst_1.id())
    param = md.md
    print(param.getName())
    print(param.getValue())    

    md = marketDataRepositorySingleton.getMarketData(inst_1.type())
    print(md)
