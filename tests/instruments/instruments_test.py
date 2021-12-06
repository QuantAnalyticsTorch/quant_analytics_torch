# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.instruments import instruments

def test_instruments():
    
    es = instruments.Asset("SPX")
    assert es.type() == "Asset"

if __name__ == '__main__':
    test_instruments()