import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 5])
Y = np.array([0.7, 1.5, 4.5, 9.5])



W = 0
b = 0

def linear(W, b, x):
    return W * x + b

def dW_linear(W, b, x):
    return x

def db_linear(W, b, x):
    return 1

def dx_linear(W, b, x):
    return W

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def da_sigmoid(a):
    return np.exp(-a)/(1+(np.exp(-a)**2))

def model(W, b, x):
    return sigmoid(linear(W, b, x)) * 20.0

def dW_model(W, b, x):
    return da_sigmoid(linear(W, b, x))*x

def db_model(W, b, x):
    return da_sigmoid(linear(W, b, x))

def loss(y, y_prim):
    return np.mean((y - y_prim) ** 2)

def dW_loss(y, y_prim, W, b, x): # derivative WRT Loss function
    return -2 * np.mean(x * (y - y_prim))

def db_loss(y, y_prim, W, b, x):
    return -2 * np.mean(y - y_prim)


learning_rate = 1e-2
losses = []
for epoch in range(300):

    Y_prim = model(W, b, X)
    print(Y_prim)
    print(Y)
    loss_next = loss(Y, Y_prim)
    losses.append(loss_next)

    dW_loss_next = dW_loss(Y, Y_prim, W, b, X)
    db_loss_next = db_loss(Y, Y_prim, W, b, X)

    W -= dW_loss_next * learning_rate
    b -= db_loss_next * learning_rate

    print(f'Y_prim: {Y_prim}')
    print(f'loss: {loss_next}')
    print(f'W: {W}')
    print(f'b: {b}')

plt.plot(losses)
plt.title('loss')
plt.show()
