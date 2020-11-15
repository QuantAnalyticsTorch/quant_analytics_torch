import torch
import matplotlib
import matplotlib.pyplot as plt

from quant_analytics_torch.modules.longshorttermmemory import LongShortTermModule

def entropy_loss(x,d=20):
    return 1./d*torch.log(torch.mean(torch.exp(-d*x)))

if __name__ == '__main__':
    print('Quantitative analyst deep learning')

    batch_size = 1000

    model = LongShortTermModule(batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 20
    training_size = 50
    seq_len = 2
    input_dim = 1
    risk_aversion = 10
    sigma = 0.2

    for i in range(epochs):
        for j in range(training_size):

            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, model.batch_size, model.hidden_layer_size),
                        torch.zeros(1, model.batch_size, model.hidden_layer_size))

            # Initial model price is 0
            x0 = torch.zeros(batch_size, seq_len + 1, input_dim)

            # One step simulation
            dx = torch.randn(batch_size, seq_len, input_dim)

            for k in range(seq_len):
                x0[:,k+1,:] = x0[:,k,:] + sigma * dx[:,k,:]

            delta = model(x0[:,:-1,:])

            # Payoff being pi = max(dx-0,0)
            payoff = torch.max(x0[:,-1,:],torch.tensor(0.))

            # Hedge performance pnl = pi - delta * dx
            
            pnl = torch.zeros(batch_size, input_dim)
            delta_zero = torch.zeros(batch_size, input_dim)

            for k in range(seq_len):
                pnl = pnl + (delta[:,k,:]-delta_zero)  * x0[:,k,:]
                delta_zero = delta[:,k,:]

            pnl = pnl + payoff - delta[:,-1,:]*x0[:,-1,:]

            # Loss function is a mean variance optimisation
            loss = -torch.mean(pnl) + risk_aversion * torch.norm(pnl)

            # Propagate derivates
            loss.backward()
            # Optimize
            optimizer.step()

        # Print some deltas
        print(delta[0:5,:,0])

        print(f'Epoch number: {i:3} mean variance loss: {loss.item():10.6f}')

    pnl_plot = pnl[:,0].detach().numpy()
    payoff_plot = payoff[:,0].detach().numpy()

    plt.hist([pnl_plot,payoff_plot])