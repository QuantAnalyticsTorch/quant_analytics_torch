import numpy as np
from scipy import integrate
from quant_analytics_torch.analytics.blackanalytics import impliedvolatility
from scipy.special import gamma
from mpmath import *

def bivariatecharacteristicfunction_heston(u,ld,x,v,kappa,theta,eps,rho,dt):
    gamma = np.sqrt((kappa-1j*eps*rho*u)*(kappa-1j*eps*rho*u) + eps*eps*(1j*u+u*u+2*ld))
    alpha = gamma + kappa - 1j*eps*rho*u
    beta = gamma - kappa + 1j*eps*rho*u
    a = kappa*theta/eps/eps*((kappa-gamma-1j*eps*rho*u)*dt-2.*np.log((alpha+beta*np.exp(-gamma*dt))/(alpha+beta)))
    b = ((1j*u+u*u+2*ld)*(np.exp(gamma*dt)-1))/((gamma+kappa-1j*eps*rho*u)*(np.exp(gamma*dt)-1)+2*gamma)
    return np.exp(1j*u*x+a-b*v)

def heston_option_price(strike,forward,v,kappa,theta,eps,rho,dt):
    x = np.log(forward)
    k = np.log(strike)
    ad = 0.5
    f = lambda u: bivariatecharacteristicfunction_heston(u-(ad+1)*1j,0,x,v,kappa,theta,eps,rho,dt)*np.exp(-1j*u*k)/(-(u-1j*ad)*(u-1j*(ad+1)))
    y = integrate.quad(f, 0, 20, args=())
    return y[0]*np.exp(-ad*k)/np.pi


def bivariatecharacteristicfunction_threeovertwo(u,ld,x,v,kappa,theta,eps,rho,dt):
    """
    Bivariate characteristic function in the 3/2 model
        $$ \phi(u,\lambda,x,v) = \left[\left. e^{i u X_T - \lambda \int_t^T v_s d s}\\right| x_t=x, v_t=v \\right] $$
    """
    p = -kappa + 1j*eps*rho*u
    q = ld + 1j*u/2 + u*u/2
    alpha = -(1/2-p/eps/eps)+np.sqrt((1/2-p/eps/eps)**2+2*q/eps/eps)
    g = 2*(alpha+1-p/eps/eps)
    y = v*(np.exp(kappa*theta*dt)-1)/(kappa*theta)
    ret = fp.hyp1f1(alpha,g,(-2)/(eps*eps*y))
    return np.exp(1j*u*x)*gamma(g-alpha)/gamma(g)*((2)/(eps*eps*y))**alpha*ret

def threeovertwo_option_price(strike,forward,v,kappa,theta,eps,rho,dt):
    x = np.log(forward)
    k = np.log(strike)
    ad = 0.5
    f = lambda u: bivariatecharacteristicfunction_threeovertwo(u-(ad+1)*1j,0,x,v,kappa,theta,eps,rho,dt)*np.exp(-1j*u*k)/(-(u-1j*ad)*(u-1j*(ad+1)))
    y = fp.quad(f, [0, 50], args=())
    return float(y.real)*np.exp(-ad*k)/np.pi

def threeovertwo_vix_price(strike,forward,v,kappa,theta,eps,rho,dt):
    """
    Using the Laplace transform of the 3/2 model
        $$ \phi(\lambda,v) = \left[\left. e^{\lambda \int_t^T v_s d s}\\right| v_t=v \\right] $$
    """
    x = np.log(forward)
    k = np.log(strike)
    ad = 0.5
    f = lambda u: bivariatecharacteristicfunction_threeovertwo(u-(ad+1)*1j,0,x,v,kappa,theta,eps,rho,dt)*np.exp(-1j*u*k)/(-(u-1j*ad)*(u-1j*(ad+1)))
    y = fp.quad(f, [0, 50], args=())
    return float(y.real)*np.exp(-ad*k)/np.pi

if __name__ == '__main__':
    strikes = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ivs_heston = []
    ivs_3_2 = []    

    for it in strikes:
        strike = np.exp(it)
        price = heston_option_price(strike,1,0.04,1,0.04,1,0,1)
        iv = impliedvolatility(price,1,strike,1)
        #ivs_heston.append(iv)

    for it in strikes:
        strike = np.exp(it)
        price = threeovertwo_option_price(strike,1,0.04,22,0.09,8,-0.8,1)
        #iv = impliedvolatility(price.real,1,strike,1)
        #ivs_3_2.append(iv)

    print(ivs_heston)
    print(ivs_3_2)
