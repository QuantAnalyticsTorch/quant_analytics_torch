# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.interpolators import interpolate
from quant_analytics_torch.models import models, modelcomponentbase

import torch
import datetime

class DynamicModelComponent(modelcomponentbase.ModelComponentBase):
    def __init__(self):
        super().__init__()

class LognormalModelComponent(DynamicModelComponent):
    def __init__(self):
        super().__init__()

    def dim(self):
        return 1