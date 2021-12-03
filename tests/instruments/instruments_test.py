from quant_analytics_torch.instruments import instruments

def test_instruments():
    
    es = instruments.EquitySpot("SPX")
    assert es.type() == "EquitySpot"

if __name__ == '__main__':
    test_instruments()