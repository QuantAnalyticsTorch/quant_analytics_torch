# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.instruments import instruments, currencies
from quant_analytics_torch.marketdata import marketdata
from quant_analytics_torch.analytics import constants

import torch

def test_marketdata():
    inst = instruments.Asset("SPX", currencies.USD)
    md = marketdata.MarketData(inst, 100., { 'ccy' : "EUR" })
    param = md.md
    v = param.getValue()

    w = v*v
    w.backward()
    dwdv = v.grad

    assert abs(dwdv - 200.) < constants.EPSILON


if __name__ == '__main__':
    test_marketdata()