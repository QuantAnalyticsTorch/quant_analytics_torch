# quant_analytics_torch
Quantitative analytics with Pytorch and Deep learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuantAnalyticsTorch/quant_analytics_torch/blob/main/docs/source/examples/SSVICalibration.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantAnalyticsTorch/quant_analytics_torch.git/main?urlpath=lab%2Ftree%2Fdocs%2Fsource%2Fexamples)

Further documentation https://quantanalyticstorch.github.io/

```bash
pip install quant-analytics-torch
```

mkdir docs
cd docs
sphinx-quickstart
sphinx-apidoc -o source/ ../quant_analytics_torch
make html

## Run the tests

pytest --cov-report term --cov=quant_analytics_torch tests/ --html=./test-reports/report.html --cov-report=html:./test-reports/coverage --profile