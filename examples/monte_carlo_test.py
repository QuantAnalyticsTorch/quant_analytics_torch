import torch
import time
import numpy as np

number =  20000000

SQRT_2 = np.sqrt(2.)

torch.set_num_threads(1)

def norminv(x):
    return torch.erfinv(2.*(x-0.5))*SQRT_2

time_start = time.time()

soboleng = torch.quasirandom.SobolEngine(dimension=1)
z = soboleng.draw(number)
w = norminv(z)

x = torch.tensor(1.0, requires_grad = True)
sigma = torch.tensor(0.2, requires_grad = True)
variancehalf = -sigma*sigma/2
k = torch.tensor(1.0)

s = x * torch.exp(variancehalf + sigma*w)

v = torch.mean(torch.max(s-k,torch.tensor(0.0)))

v.backward()

time_end = time.time()
time_price_cpu = time_end - time_start
print("Runtime on CPU: ", time_price_cpu)