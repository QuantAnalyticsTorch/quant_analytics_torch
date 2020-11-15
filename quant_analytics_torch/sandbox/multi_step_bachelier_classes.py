import torch
import matplotlib
import matplotlib.pyplot as plt

from quant_analytics_torch.calculators.deeplearninghedgingengine import DeepLearningValuationEngine

if __name__ == '__main__':
    print('Quantitative analyst deep learning')

    deepLearningValuation = DeepLearningValuationEngine()
    deepLearningValuation.run()

    paths = 10000

    x0 = deepLearningValuation.paths(paths)

    # Payoff being pi = max(dx-0,0)
    payoff = deepLearningValuation.payoff(x0)

    deepLearningValuation.init_hidden(paths)

    # Hedge performance pnl = pi - delta * dx
    delta = deepLearningValuation.delta(x0)

    # Generate the pnl
    pnl = deepLearningValuation.pnl(x0, delta, payoff, paths)

    pnl_plot = pnl[:,0].detach().numpy()
    payoff_plot = payoff[:,0].detach().numpy()

    plt.hist([pnl_plot,payoff_plot])