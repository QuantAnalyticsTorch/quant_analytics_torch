# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.models import models

import torch

class BaseCalculator(torch.nn.Module):
    def __init__(self, instrument : instruments.InstrumentBase, model : models.Model):
        super().__init__()
        self.instrument = instrument
        self.model = model

    def calculate(self):
        return None