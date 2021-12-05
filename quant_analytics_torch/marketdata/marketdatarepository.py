# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdata
from quant_analytics_torch.instruments import instruments

import datetime


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
    inst_1 = instruments.Asset("SPX")
    inst_2 = instruments.Asset("AAPL")

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


    fwd_1 = instruments.Forward("SPX-1", inst_1, datetime.datetime(2021,12,12) )
    fwd_2 = instruments.Forward("SPX-2", inst_1, datetime.datetime(2022,12,12) )
    fwd_3 = instruments.Forward("SPX-3", inst_1, datetime.datetime(2023,12,12) )

    fwd = [fwd_1, fwd_2, fwd_3]

    for f in fwd:
        md = marketdata.MarketData(f,100.)
        marketDataRepositorySingleton.storeMarketData(md)

    md = marketDataRepositorySingleton.getMarketData( { fwd_1.type() : inst_1.name } )

    print(md)