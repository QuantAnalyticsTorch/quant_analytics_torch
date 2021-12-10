import torch

class BaseProduct(torch.nn.Module):
    def __init__(self):
        super(BaseProduct).__init__()
        self.data = data
        self.dates_underylings = {}

    def getDatesUnderlying(self):
        return self.dates_underylings

    def productData(self):
        return productdata.ProductDataBase(self.dates_underylings)

    def getNumberOfLegs(self):
        return 0

class InterestRateBaseProduct(BaseProduct):
    def __init__(self):
        super(BaseInterestRateProduct, self).__init__(data)

class SingleCashflow(BaseProduct):
    def __init__(self):
        super(BaseInterestRateProduct, self).__init__(data)
        self.expiry = self.data['expiry']
        self.ccy = self.data['ccy']

    def getExpiry(self):
        return self.expiry

    def getCcy(self):
        return self.ccy
