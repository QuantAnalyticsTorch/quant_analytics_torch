from quant_analytics_torch.analytics import characteristicfunction, blackanalytics

def test_blackscholes():
    p = characteristicfunction.blackscholes_option_price(1.,1.,0.2,1.)
    v = blackanalytics.black(1.,1.,1.,0.2,0.0)
    assert abs(p-v) < 0.01


if __name__ == '__main__':
    test_blackscholes()