import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from quant_analytics_torch.modules.longshorttermmemory import LongShortTermModule

def entropy_loss(x,d=20):
    return 1./d*torch.log(torch.mean(torch.exp(-d*x)))

if __name__ == '__main__':
    print('Quantitative analyst deep learning')

    training_size = 20
    batch_size = 10**3
    simulations = training_size * batch_size

    model = LongShortTermModule(batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    seq_len = 2
    input_dim = 1
    risk_aversion = 10
    sigma = 0.2


    # Initial model price is 0
    x0 = torch.zeros(simulations, seq_len + 1, input_dim)

    # One step simulation
    dx = torch.randn(simulations, seq_len, input_dim)

    for k in range(seq_len):
        x0[:,k+1,:] = x0[:,k,:] + sigma * dx[:,k,:]


    for i in range(epochs):
        model.train()

        for j in range(training_size):

            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, model.batch_size, model.hidden_layer_size),
                        torch.zeros(1, model.batch_size, model.hidden_layer_size))

            delta = model(x0[j*batch_size:(j+1)*batch_size,:-1,:])

            # Payoff being pi = max(dx-0,0)
            payoff = torch.max(x0[j*batch_size:(j+1)*batch_size,-1,:],torch.tensor(0.))

            # Hedge performance pnl = pi - delta * dx
            
            pnl = torch.zeros(batch_size, input_dim)
            delta_zero = torch.zeros(batch_size, input_dim)

            for k in range(seq_len):
                pnl = pnl + (delta[:,k,:]-delta_zero)  * x0[j*batch_size:(j+1)*batch_size,k,:]
                delta_zero = delta[:,k,:]

            pnl = pnl + payoff - delta[:,-1,:]*x0[j*batch_size:(j+1)*batch_size,-1,:]

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