# Copyright (c) Quant Analytics. All rights reserved.
from dataclasses import dataclass

@dataclass
class Currency(object):
    ccy : str

    def toString(self):
        return self.ccy

USD = Currency("USD")
EUR = Currency("USD")

if __name__ == '__main__':
    ccy = Currency("JPY")
    print(ccy.toString())

    ccyj = ccy.to_json()
    print(ccyj)