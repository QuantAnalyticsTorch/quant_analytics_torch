# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies

import datetime

def test_instruments():
    
    es = instruments.Asset("SPX", currencies.USD)
    assert es.type() == "Asset"

    cd = instruments.CashDeposit("USD-1y", currencies.USD, datetime.datetime(2022,12,12))
    assert cd.type() == "CashDeposit"

if __name__ == '__main__':
    test_instruments()