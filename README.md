# quant_analytics_torch

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/Ph2XUS4N8g)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuantAnalyticsTorch/quant_analytics_torch/blob/main/docs/source/examples/SSVICalibration.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantAnalyticsTorch/quant_analytics_torch.git/main?urlpath=lab%2Ftree%2Fdocs%2Fsource%2Fexamples)

Quantitative analytics with Pytorch and Deep learning

```bash
pip install quant-analytics-torch
```

Further documentation https://quant-analytics-torch.readthedocs.io/

## Code development

Repo is setup for development via Github codespaces

## Contribution

If you like to discuss this repo, join me on Discord

https://discord.gg/U2VReK4m

## Run the tests

pytest --cov-report term --cov=quant_analytics_torch tests/ --html=./test-reports/report.html --cov-report=html:./test-reports/coverage --profile

## Some useful comments

mkdir docs
cd docs
sphinx-quickstart
sphinx-apidoc -o source/ ../quant_analytics_torch
make html
