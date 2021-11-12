import torch
import time

import math
import numpy as np

SQRT_2 = np.sqrt(2.)


class UnivariateBrownianBridge():
    def __init__(self, number_time_steps):
        self.number_time_steps = number_time_steps

        self.left_index = torch.zeros(number_time_steps, dtype=int)
        self.right_index = torch.zeros(number_time_steps, dtype=int)
        self.bridge_index = torch.zeros(number_time_steps, dtype=int)
        self.left_weight = torch.zeros(number_time_steps)
        self.right_weight = torch.zeros(number_time_steps)
        self.std_dev = torch.zeros(number_time_steps)

        self._map = torch.zeros(number_time_steps, dtype=int)

        self._map[-1] = 1
        self.bridge_index[0] = number_time_steps - 1
        self.std_dev[0] = torch.sqrt(torch.tensor(1.0) * number_time_steps)
        self.left_weight[0] = 0
        self.right_weight[0] = 0

        j=0
        for i in range(1,number_time_steps):
            while self._map[j] == True:
                j = j + 1
            k = j
            while self._map[k] == False:
                k = k + 1
            l = j+((k-1-j)>>1)
            self._map[l]=i
            self.bridge_index[i]=l
            self.left_index[i]=j
            self.right_index[i]=k
            self.left_weight[i]=(k-l)/(k+1-j)
            self.right_weight[i]=(1+l-j)/(k+1-j)
            self.std_dev[i]=np.sqrt(((1+l-j)*(k-l))/(k+1-j))
            j=k+1
            if j>=number_time_steps:
                j=0

    @torch.jit.script
    def _buildPath(path, z, increment: bool, number_time_steps: int, left_index, right_index, bridge_index, left_weight, right_weight, std_dev):
        path[-1] = std_dev[0]*z[0]
        j = 0
        k = 0
        l = 0
        i = 0
        for i in range(1,number_time_steps):
            j = left_index[i]
            k = right_index[i]
            l = bridge_index[i]
            lw = left_weight[i]
            rw = right_weight[i]
            sd = std_dev[i]
            if j > 0:
                path[l] = path[j-1] * lw + path[k] * rw + z[i] * sd
            else:
                path[l] = right_weight[i] * path[k] + std_dev[i] * z[i]

        if increment:
            for i in range(1, number_time_steps):
                path[-i] = path[-i] - path[-(i+1)]

    def path(self, path, z, increment):
        return self._buildPath(path, z, increment, self.number_time_steps, self.left_index, self.right_index, self.bridge_index, self.left_weight, self.right_weight, self.std_dev)