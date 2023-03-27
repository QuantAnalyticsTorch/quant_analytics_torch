from quant_analytics_torch.analytics import constants
import torch

def hyperbolic(x : torch.tensor) -> torch.tensor:
    """ Using the hyperbolic function
    .. _target hyperbolic:

    .. math::

        f(x) = \\frac{1}{2} \\left(x + \sqrt{1 + x^2} \\right)

    Args:
        x (torch.tensor): M-dimensional tensor

    Returns:
        y (torch.tensor): Hyperbolic function
    """

    return (x + torch.sqrt(1. + x*x))/2.

def hyperbolic_prime(x : torch.tensor) -> torch.tensor:
    """ Using the derivative of the :hoverxref:`hyperbolic function <target hyperbolic>`
    
    .. _target hyperbolic_prime:

    .. math::

        f(x) = \\frac{1}{2}\\left(1 + \\frac{x}{\sqrt{1 + x^2}} \\right)

    Args:
        x (torch.tensor): M-dimensional tensor

    Returns:
        y (torch.tensor): Derivative of hyperbolic function
    """
    return (1 + x/torch.sqrt(1. + x*x))/2.

def soft_max_hyperbolic(x,eps=constants.EPSILON):
    return hyperbolic(x/eps)*eps

def soft_heavy_side_hyperbolic(x,eps=constants.EPSILON):
    return hyperbolic_prime(x/eps)