# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdata
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies

import datetime
import sys


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

    def __getitem__(self, idx):        
        md = self.data
        while isinstance(idx, dict):
            k = next(iter(idx))
            md = md[k]
            idx = idx[k]
        return md[idx]

    def clear(self):
        self.data = {}

    def storeMarketData(self, marketdata : marketdata.MarketData):
        recursive_insert(self.data, marketdata.inst.id(), marketdata )

marketDataRepositorySingleton = MarketDataRepository()

def fillSampleDate(marketData : MarketDataRepository):
    inst_1 = instruments.Asset("SPX", currencies.USD)
    inst_2 = instruments.Asset("AAPL", currencies.USD)

    md_1 = marketdata.MarketData(inst_1, 100.)
    marketData.storeMarketData(md_1)

    md_2 = marketdata.MarketData(inst_2, 100.)
    marketData.storeMarketData(md_2)

    fwds = []
    cashdeposits = []

    fwd_dates = [datetime.datetime(2021,12,12), datetime.datetime(2022,12,12), datetime.datetime(2023,12,12)]
    fva = [101., 102., 103.]    
    dfs = [0.99, 0.98, 0.97]        

    for j,jt in enumerate(fwd_dates):
        fwds.append( instruments.Forward("SPX-" + str(j), inst_1, jt ) )
        cashdeposits.append( instruments.CashDeposit("USD-" + str(j), currencies.USD, jt ) )        

    for i,f in enumerate(fwds):
        md = marketdata.MarketData(f,fva[i])
        marketData.storeMarketData(md)

    for i,f in enumerate(cashdeposits):
        md = marketdata.MarketData(f,dfs[i])
        marketData.storeMarketData(md)


    # Get the cash deposit data
    ssviVolatilityTheta = instruments.SSVIVolatility("SSVI-1y-Theta", inst_1, datetime.datetime(2022,12,12), instruments.SSVIParam.Theta)
    md_theta = marketdata.MarketData(ssviVolatilityTheta, 0.2)    
    marketDataRepositorySingleton.storeMarketData(md_theta)

    ssviVolatilityBeta = instruments.SSVIVolatility("SSVI-1y-Beta", inst_1, datetime.datetime(2022,12,12), instruments.SSVIParam.Beta)
    md_beta = marketdata.MarketData(ssviVolatilityBeta, 3.0)    
    marketDataRepositorySingleton.storeMarketData(md_beta)

    ssviVolatilityRho = instruments.SSVIVolatility("SSVI-1y-Rho", inst_1, datetime.datetime(2022,12,12), instruments.SSVIParam.Rho)
    md_rho = marketdata.MarketData(ssviVolatilityRho, -0.5)    
    marketDataRepositorySingleton.storeMarketData(md_rho)

    ssviVolatilityTheta = instruments.SSVIVolatility("SSVI-2y-Theta", inst_1, datetime.datetime(2023,12,12), instruments.SSVIParam.Theta)
    md_theta = marketdata.MarketData(ssviVolatilityTheta, 0.2)    
    marketDataRepositorySingleton.storeMarketData(md_theta)

    ssviVolatilityBeta = instruments.SSVIVolatility("SSVI-2y-Beta", inst_1, datetime.datetime(2023,12,12), instruments.SSVIParam.Beta)
    md_beta = marketdata.MarketData(ssviVolatilityBeta, 3.0)    
    marketDataRepositorySingleton.storeMarketData(md_beta)

    ssviVolatilityRho = instruments.SSVIVolatility("SSVI-2y-Rho", inst_1, datetime.datetime(2023,12,12), instruments.SSVIParam.Rho)
    md_rho = marketdata.MarketData(ssviVolatilityRho, -0.5)    
    marketDataRepositorySingleton.storeMarketData(md_rho)


if __name__ == '__main__':
    fillSampleDate(marketDataRepositorySingleton)
    inst_1 = instruments.Asset("SPX", currencies.USD)
    inst_2 = instruments.Asset("AAPL", currencies.USD)

    md = marketDataRepositorySingleton[inst_1.id()]
    param = md.md
    print(param.getName())
    print(param.getValue())    

    md = marketDataRepositorySingleton[inst_1.type()]
    print(md)

    fwd_1 = instruments.Forward("SPX-1", inst_1, datetime.datetime(2021,12,12) )

    md = marketDataRepositorySingleton[{ fwd_1.type() : inst_1.name }]

    print(md)

    # Get the cash deposit data
    cashdeposit = instruments.CashDeposit(None, currencies.USD, datetime.datetime.now() )

    md = marketDataRepositorySingleton[{ cashdeposit.type() : cashdeposit.ccy.toString() }]

    print(md)

    # Get the cash deposit data
    ssviVolatility = instruments.SSVIVolatility(None, inst_1)

    md_ssvi = marketDataRepositorySingleton[{ ssviVolatility.type() : inst_1.name }]

    print(md_ssvi)
