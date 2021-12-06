# Copyright (c) Quant Analytics. All rights reserved.
class Currency(object):
    def __init__(self, ccy = str()):
        super().__init__()
        self.ccy = ccy

    def toString(self):
        return self.ccy

USD = Currency("USD")
EUR = Currency("USD")

if __name__ == '__main__':
    ccy = Currency("JPY")
    print(ccy.toString())