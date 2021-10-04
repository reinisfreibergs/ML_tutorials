import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# y' = 2x
A = 0 #robeznosacijumi
Model = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=50),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=50, out_features=1)
        )

def Psi_t(x): # atrisinajuma forma
    return A + x * Model(x)

def analytical_derivative(x, function): #no funkcijas izteikts pirmas kartas atvasinajums
    return 2 * x

def loss(x):
    x.requires_grad = True
    outputs = Psi_t(x)
    gradient = torch.autograd.grad(outputs= outputs,
                                   inputs= x,
                                   grad_outputs=torch.ones_like(outputs),
                                   create_graph=True)[0]
    return torch.mean( (gradient - analytical_derivative(x, outputs))  ** 2)


optimizer = torch.optim.LBFGS(Model.parameters())
x = torch.Tensor(np.linspace(0, 2, 100)[:, None])

def closure():

    optimizer.zero_grad()
    l = loss(x)
    l.backward()
    return l

for epoch in range(10):
    optimizer.step(closure)


xx = np.linspace(0, 2, 100)[:, None]
with torch.no_grad():
    y_NN = Psi_t(torch.Tensor(xx)).numpy()
y_analytical = xx**2 # analitiskais atrisinajums

fig, ax = plt.subplots(dpi=100)
ax.plot(xx, y_analytical, label='True')
ax.plot(xx, y_NN, '--', label='Neural network approximation')
ax.set_xlabel('$x$')
ax.set_ylabel('$Psi(x)$')
plt.legend(loc='best');
plt.show()
