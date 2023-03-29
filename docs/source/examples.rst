.. _examples:

.. Quant Analytics Flow documentation master file, created by
   sphinx-quickstart on Mon Dec  7 20:22:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Examples
========================================================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   ./examples/SSVICalibration
   ./examples/EquityCashDividends
   ./examples/HestonQuadraticExponentialSimulation
   ./examples/32QuadraticExponentialSimulation

Example notebooks

.. grid:: 2
    :margin: 5 5 0 0
    :gutter: 4

    .. grid-item-card:: SSVI Calibration
        :link: ./examples/SSVICalibration
        :link-type: doc

        Example of SSVI calibration

    .. grid-item-card:: Equity cash dividends
        :link: ./examples/EquityCashDividends
        :link-type: doc

        Example of equity cash dividend volatility conversion

    .. grid-item-card:: Heston Quadratic Exponential Simulation
        :link: ./examples/HestonQuadraticExponentialSimulation
        :link-type: doc

        Pytorch (vectorized) Heston implementation of the Quadratic Exponential scheme      

    .. grid-item-card:: 32 Quadratic Exponential Simulation
        :link: ./examples/32QuadraticExponentialSimulation
        :link-type: doc

        Pytorch (vectorized) 3/2 implementation of the Quadratic Exponential scheme