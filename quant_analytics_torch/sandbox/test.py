import torch

class BaseDeepLearning(torch.nn.Module):
    def __init__(self):
        super(BaseDeepLearning, self).__init__()

        n_in = 2
        n_out = 1
        n_layers = 1
        n_hidden = 512


        self.regressor = torch.nn.Sequential(torch.nn.Linear(n_in, n_hidden),
                                       torch.nn.ReLU(inplace=True),
                                       torch.nn.Linear(n_hidden, n_hidden),
                                       torch.nn.ReLU(inplace=True),
                                       torch.nn.Linear(n_hidden, n_out))

    def forward(self, x):
        output = self.regressor(x)
        return output


    def simulate_pnl(self, simulations, times, states, payoffs):
        pnl = torch.zeros(simulations)

        state = states[0]
        time = times[0]*torch.ones(simulations)
        x = torch.stack([state, time],1)
        delta = self.forward(x)[:,0]

        # Start by buying delta of shares
        pnl = delta * states[0]

        for i,it in enumerate(times[1:]):
            print(i)

        pnl += payoffs - delta * states[-1]

        return pnl

class BaseLSTMDeepLearning(torch.nn.Module):
    def __init__(self):
        super(BaseLSTMDeepLearning, self).__init__()

        input_dim = 2
        hidden_dim = 10
        n_layers = 1

        batch_size = 1
        seq_len = 1

        self.hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
        self.cell_state = torch.randn(n_layers, batch_size, hidden_dim)
        self.hidden = (self.hidden_state, self.cell_state)

        self.regressor = torch.nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)


    def forward(self, x):
        output, self.hidden = self.regressor(x,self.hidden)
        return output


    def simulate_pnl(self, simulations, times, states, payoffs):
        pnl = torch.zeros(simulations)

        state = states[0]
        time = times[0]*torch.ones(simulations)
        x = torch.stack([state, time],1)
        delta = self.forward(states)

        # Start by buying delta of shares
        pnl = delta * states[0]

        for i,it in enumerate(times[1:]):
            print(i)

        pnl += payoffs - delta * states[-1]

        return pnl


class Simulator(object):
    def __init__(self):
        super(Simulator, self).__init__()

        self.base_deep_learning = BaseDeepLearning()
        #self.base_deep_learning = BaseLSTMDeepLearning()

    def loss(self, pnl):
        lam = torch.tensor(0.2)
        return 1/lam*torch.mean(torch.exp(-lam*pnl))


    def optimise(self):

        n_epochs = 20
        n_epoch_size = 20

        optimizer = torch.optim.Adam(self.base_deep_learning.parameters(), lr=0.00001)

        times = [1.0]

        nt = len(times)

        for epoch in range(n_epochs):
            self.base_deep_learning.train()

            nc = 10000
            epsilon = 0.0
            sigma = 0.2

            S = torch.zeros(size=(nt+1,nc))

            dW = torch.randn(nt+1,nc)

            S0 = 1.0
            k = 1.0

            S[0] = S0 *(1 + epsilon*dW[0])

            for i,it in enumerate(times):
                S[i+1] = S[i] * (1 + sigma*dW[i+1])

            payoff = torch.max(S[-1]-k,torch.tensor(0.0))

            print("option value="+str(torch.mean(payoff)))

            for epoch_size in range(n_epoch_size):

                optimizer.zero_grad()
                pnl = self.base_deep_learning.simulate_pnl(nc, times, S, payoff)  
                loss = self.loss(pnl)

                loss.backward()
                optimizer.step()

            pnl = self.base_deep_learning.simulate_pnl(nc, times, S, payoff)
            print("pnl="+str(torch.mean(pnl)))
            #print(pnl[0:10])
            loss = self.loss(pnl)

            print("loss="+str(loss))

class LSTMSimulator(object):
    def __init__(self):
        super(Simulator, self).__init__()

        self.base_deep_learning = BaseDeepLearning()
        #self.base_deep_learning = BaseLSTMDeepLearning()

    def loss(self, pnl):
        lam = torch.tensor(0.5)
        return 1/lam*torch.mean(torch.exp(-lam*pnl))


    def optimise(self):

        n_epochs = 20
        n_epoch_size = 20

        optimizer = torch.optim.Adam(self.base_deep_learning.parameters(), lr=0.00001)

        times = [1.0]

        nt = len(times)

        for epoch in range(n_epochs):
            self.base_deep_learning.train()

            nc = 10000
            epsilon = 0.0
            sigma = 0.2

            S = torch.zeros(size=(nt+1,nc))

            dW = torch.randn(nt+1,nc)

            S0 = 1.0
            k = 1.0

            S[0] = S0 *(1 + epsilon*dW[0])

            for i,it in enumerate(times):
                S[i+1] = S[i] * (1 + sigma*dW[i+1])

            payoff = torch.max(S[-1]-k,torch.tensor(0.0))

#            print("option value="+str(torch.mean(payoff)))

            for epoch_size in range(n_epoch_size):

                optimizer.zero_grad()
                pnl = self.base_deep_learning.simulate_pnl(nc, times, S, payoff)  
                loss = self.loss(pnl)

                loss.backward()
                optimizer.step()

            pnl = self.base_deep_learning.simulate_pnl(nc, times, S, payoff)
#            print("pnl="+str(torch.mean(pnl)))
            #print(pnl[0:10])
            loss = self.loss(pnl)

#            print("loss="+str(loss))



if __name__ == '__main__':
    print('Quantitative analyst deep learning')

    simulator = Simulator()
    simulator.optimise()

