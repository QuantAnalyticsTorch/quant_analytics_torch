from quant_analytics_torch.calculators.univariatebrownianbridge import UnivariateBrownianBridge

import torch

def square_root_symmetric_matrix(A):
    w, v = torch.linalg.eigh(A)
    return torch.mm(torch.mm(v, torch.diag(torch.sqrt(w[:]))), v.t())


class MultivariateBrownianBridge():
    def __init__(self, forwardCovarianceMatrices):
        self.forwardCovarianceMatrices = forwardCovarianceMatrices
        self.numberTimeSteps = len(forwardCovarianceMatrices)
        self.numberStates = len(forwardCovarianceMatrices[0])
        self.brownian = UnivariateBrownianBridge(self.numberTimeSteps)
        self.sqrtForwardCovarianceMatrices = torch.zeros(size=(self.numberTimeSteps, self.numberStates, self.numberStates))
        for i in range(self.numberTimeSteps):
            self.sqrtForwardCovarianceMatrices[i] = square_root_symmetric_matrix(self.forwardCovarianceMatrices[i])

    def path(self, z, increments):
        path = torch.zeros(size=(self.numberTimeSteps, self.numberStates, len(z[0][0])))
        self.brownian.path(path, z, increments)

        result = torch.zeros(size=(self.numberTimeSteps, self.numberStates, len(z[0][0])))        
        for i in range(self.numberTimeSteps):
            result[i] = torch.matmul(self.sqrtForwardCovarianceMatrices[i], path[i])
        return result