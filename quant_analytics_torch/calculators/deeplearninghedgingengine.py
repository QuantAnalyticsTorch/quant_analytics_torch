import torch

from quant_analytics_torch.modules.longshorttermmemory import LongShortTermModule
from quant_analytics_torch.calculators.pathgeneratornormal import PathGeneratorNormal

class DeepLearningValuationEngine(torch.nn.Module):
    def __init__(self, input_size=1, batch_size=1000, hidden_layer_size=128, output_size=1, epochs=20, training_size=15, seq_len = 2):
        super().__init__()

        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.epochs = epochs
        self.training_size = training_size
        self.seq_len = seq_len

        self.model = LongShortTermModule(batch_size=self.batch_size, hidden_layer_size=self.hidden_layer_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        self.input_dim = 1
        self.risk_aversion = 20
        self.sigma = 0.2

        self.pathgenerator = PathGeneratorNormal(seq_len=self.seq_len, forwardvariance=torch.tensor(0.04), fwd=0.0)


    def init_hidden(self, batch_size):
        self.model.hidden_cell = (  torch.zeros(1, batch_size, self.model.hidden_layer_size),
                                    torch.zeros(1, batch_size,  self.model.hidden_layer_size))

    def paths(self, batch_size):
        return self.pathgenerator.paths(batch_size)

    def payoff(self, x0):
        return torch.max(x0[:,-1,:],torch.tensor(0.))

    def delta(self, x0):
        return self.model(x0[:,:-1,:])

    def pnl(self, x0, delta, payoff, batch_size):
        pnl = torch.zeros(batch_size, self.input_dim)
        delta_zero = torch.zeros(batch_size, self.input_dim)

        for k in range(self.seq_len):
            pnl = pnl + (delta[:,k,:]-delta_zero)  * x0[:,k,:]
            delta_zero = delta[:,k,:]

        return pnl + payoff - delta[:,-1,:]*x0[:,-1,:]

    def run(self):

        # Total number of Monte-Carlo simulations
        simulation_size = self.batch_size * self.training_size

        # Only do one simulation at the beginning
        x0_t = self.paths(simulation_size)

        for i in range(self.epochs):

            self.model.train()

            for j in range(self.training_size):

                self.optimizer.zero_grad()

                # Reinitialise the hidden cells
                self.init_hidden(self.batch_size)

                # Take a subset of the paths
                x0 = x0_t[j*self.batch_size:(j+1)*self.batch_size,:,:]

                # Payoff being pi = max(dx-0,0)
                payoff = self.payoff(x0)

                # Hedge performance pnl = pi - delta * dx
                delta = self.delta(x0)

                # Generate the pnl
                pnl = self.pnl(x0, delta, payoff, self.batch_size)

                # Loss function is a mean variance optimisation
                loss = -torch.mean(pnl) + self.risk_aversion * torch.norm(pnl)

                # Propagate derivates
                loss.backward()
                # Optimize
                self.optimizer.step()

            # Print some deltas
            print(delta[0:5,:,0])

            print(f'Epoch number: {i:3} mean variance loss: {loss.item():10.6f}')

if __name__ == '__main__':
    print('Quantitative analyst deep learning')

    deepLearningValuation = DeepLearningValuationEngine(batch_size=1000, hidden_layer_size=256, seq_len=4)
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