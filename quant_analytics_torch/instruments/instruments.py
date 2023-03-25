# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.instruments import currencies

import datetime
from dataclasses import dataclass, field
from enum import Enum, auto

@dataclass
class InstrumentBase(object):
    """ Instrument base class """
    datesInstruments = {}

    def type(self):
        return self.__class__.__name__

    def id(self):
        return None

    def getCcy(self):
        return None

    def getMaturity(self):
        return None

    def __repr__(self):
        pass

    def payoff(self):
        pass

    def datesInstruments(self):
        return self.datesInstruments

@dataclass
class CashDeposit(InstrumentBase):
    """ Cash deposit class """
    name : str
    currency : currencies.Currency
    maturity : datetime.datetime = datetime.datetime.now()

    def id(self):
        return { self.type() : { self.currency.toString() : self.maturity } }

    def getCcy(self):
        return self.currency
    
    def getMaturity(self):
        return self.maturity


@dataclass
class Asset(InstrumentBase):
    """Asset class """
    name : str
    currency : currencies.Currency = field(default_factory=currencies.USD)

    def id(self):
        return { self.type() : { self.name : self.currency.toString() } }

    def getCcy(self):
        return self.currency

    def __get__(self, date : datetime.datetime):
        pass

@dataclass
class Forward(InstrumentBase):
    """ Forward on an underlying """
    name : str
    inst : InstrumentBase
    maturity : datetime.datetime = datetime.datetime.now()
    strike : float = float('NaN')
    currency : currencies.Currency = field(default_factory=currencies.USD)

    def id(self):
        return { self.type() : { self.inst.name : { self.maturity : self.strike } } }

    def getCcy(self):
        return self.currency

    def payoff(self):
        pass


class OptionType(Enum):
    PUT = auto()
    CALL = auto()
    STRADDLE = auto()


@dataclass
class EuropeanOption(InstrumentBase):
    """  European option """
    name : str
    inst : InstrumentBase
    maturity : datetime.datetime
    strike : float
    optiontype : OptionType = OptionType.CALL
    currency : currencies.Currency = field(default_factory=currencies.Currency)

    def id(self):
        return { self.type() : { self.inst.name : { self.maturity : { self.strike : { self.optiontype : self.currency } } } } }

    def getCcy(self):
        return self.currency

    def getMaturity(self):
        return self.maturity

#    def __get__(self, evolutionGenerator : evoluationGeneratorBase, stateTensor):
#        return None


class SSVIParam(Enum):
    Theta = auto()
    Beta = auto()
    Rho = auto()


# Also have some synthetic data
@dataclass
class SSVIVolatility(InstrumentBase):
    name : str
    inst : InstrumentBase
    maturity : datetime.datetime = datetime.datetime.now()
    paramType : SSVIParam = SSVIParam.Theta

    def id(self):
        return { self.type() : { self.inst.name : { self.maturity : self.paramType } } }

    def getCcy(self):
        return self.currency

if __name__ == '__main__':
    inst = Asset("SPX", currencies.USD)
    fwd = Forward("SPX-2011", inst, datetime.datetime(2021,12,12), 100, currencies.USD)
    option = EuropeanOption("SPX-X-Y", inst, '2021-11-11', 100., OptionType.CALL, currencies.USD)
    print(inst.type())
    print(inst.id())    

    print(fwd.type())
    print(fwd.id())    

    print(option.type())
    print(option.id())    

    cashDepo = CashDeposit("USD-1y", currencies.USD, datetime.datetime(2022,12,12))

    print(cashDepo.type())
    print(cashDepo.id())    

    print(str(fwd))

    ssviVolatilityTheta = SSVIVolatility("SSVI-1y-Theta", inst, datetime.datetime(2022,12,12), SSVIParam.Theta)
    ssviVolatilityTheta = SSVIVolatility("SSVI-2y-Theta", inst, datetime.datetime(2023,12,12), SSVIParam.Theta)    

    print(ssviVolatilityTheta.type())
    print(ssviVolatilityTheta.id())    