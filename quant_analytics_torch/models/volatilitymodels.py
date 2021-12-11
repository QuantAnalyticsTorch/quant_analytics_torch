# Copyright (c) Quant Analytics. All rights reserved.
from quant_analytics_torch.marketdata import marketdatarepository
from quant_analytics_torch.instruments import instruments
from quant_analytics_torch.instruments import currencies
from quant_analytics_torch.interpolators import interpolate
from quant_analytics_torch.models import models, modelcomponentbase

import torch
import datetime

class VolatilityModelComponent(modelcomponentbase.ModelComponentBase):
    def __init__(self, model : models.Model, marketData : marketdatarepository.MarketDataRepository, inst : instruments.InstrumentBase):
        super().__init__()
        self.param = torch.nn.ParameterList()        


class SSVIVolatilityModelComponent(VolatilityModelComponent):
    def __init__(self, model : models.Model, marketData : marketdatarepository.MarketDataRepository, inst : instruments.InstrumentBase):
        super().__init__(model, marketData, inst)
        self.param = torch.nn.ParameterList()
        self.theta_ip = None
        self.beta_ip = None
        self.rho_ip = None                

        ssvi_inst = instruments.SSVIVolatility(None, inst)
        
        # Create an index into the market data repository
        idx = { ssvi_inst.type() : inst.name }
        # Get the market data out of the market data repository
        md = marketData[idx]
        times = torch.zeros(size=(len(md),))

        # Placeholder for thetas, betas and rhos
        thetas = torch.zeros(size=(len(md),))
        betas = torch.zeros(size=(len(md),))
        rhos = torch.zeros(size=(len(md),))                

        fwds = torch.zeros(size=(len(md),))
        for i,it in enumerate(md):
            times[i] = model.dateToTime(it)
            fwd_model = model.forwards[inst.name]
            fwds[i] = fwd_model.forward(times[i])
            mdi = md[it]
            for j,jt in enumerate(mdi):
                mdj = mdi[jt]
                x = mdj.md.getValue()
                if mdj.inst.paramType == instruments.SSVIParam.Theta:
                    thetas[i] = x
                if mdj.inst.paramType == instruments.SSVIParam.Beta:
                    betas[i] = x                    
                if mdj.inst.paramType == instruments.SSVIParam.Rho:
                    rhos[i] = x
                # Append the market parameter
                self.param.append(mdj.md)            

        # We could also just store the other interpolator here
        self.forward_ip = interpolate.LinearInterpolator(times, fwds)
        self.theta_ip = interpolate.LinearInterpolator(times, thetas)
        self.beta_ip = interpolate.LinearInterpolator(times, betas)
        self.rho_ip = interpolate.LinearInterpolator(times, rhos)                

    def volatility(self, time, strike):
        fwd = self.forward_ip(time)
        theta = self.theta_ip(time)
        beta = self.beta_ip(time)
        rho = self.rho_ip(time)                
        x = torch.log(strike/fwd)
        betax = beta*x

        return theta * torch.sqrt(0.5*(1.+rho*betax + torch.sqrt(betax*betax+(1-rho)*(1-rho))))
