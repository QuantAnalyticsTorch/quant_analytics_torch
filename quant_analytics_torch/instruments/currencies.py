# Copyright (c) Quant Analytics. All rights reserved.
from dataclasses import dataclass

@dataclass
class Currency(object):
    currency : str

    def toString(self):
        return self.currency

USD = Currency("USD")
EUR = Currency("EUR")

if __name__ == '__main__':
    ccy = Currency("JPY")
    print(ccy.toString())

    ccyj = ccy.to_json()
    print(ccyj)