import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 5])
Y = np.array([0.7, 1.5, 4.5, 9.5])

W_1 = np.random.random()
b_1 = 0
W_2 = np.random.random()
b_2 = 0

def linear(W, b, x):
    return W*x + b

def relu(x):
    y = np.array(x)
    y[y<0] = 0
    return y

def model(W_1, b_1, W_2, b_2, x):
    x_hidden = relu(linear(W_1,b_1,x))
    return linear(W_2,b_2,x_hidden)

def mae(y, y_prim):
    return np.mean(np.abs(y-y_prim))

def dx_relu(x):
    y = np.ones_like(x)
    y[x<0] = 0
    return y

def dy_prim_mea(y, y_prim):
    return -(y - y_prim)/np.abs(y - y_prim)

def dW_2_mae(W_1, b_1, W_2, b_2, x, y):
    y_prim = model(W_1, b_1, W_2, b_2, x)
    return np.mean(dy_prim_mea(y, y_prim) * relu(linear(W_1,b_1,x)))

def db_2_mae(W_1, b_1, W_2, b_2, x, y):
    y_prim = model(W_1, b_1, W_2, b_2, x)
    return np.mean(dy_prim_mea(y, y_prim))

def dW_1_mae(W_1, b_1, W_2, b_2, x, y):
    y_prim = model(W_1, b_1,W_2, b_2, x)
    return np.mean(dy_prim_mea(y, y_prim) * dx_relu(linear(W_1,b_1,x)) * W_2 * x)

def db_1_mae(W_1, b_1, W_2, b_2, x, y):
    y_prim = model(W_1, b_1,W_2,b_2,x)
    return np.mean(dy_prim_mea(y, y_prim) * dx_relu(linear(W_1,b_1,x)) * W_2)

learning_rate = 1e-2
losses = []
for epoch in range(200):

    Y_prim = model(W_1, b_1, W_2, b_2, X)
    loss = mae(Y, Y_prim)
    losses.append(loss)

    dW_1 = dW_1_mae(W_1, b_1, W_2, b_2, X, Y)
    dW_2 = dW_2_mae(W_1, b_1, W_2, b_2, X, Y)

    db_1 = dW_1_mae(W_1, b_1, W_2, b_2, X, Y)
    db_2 = dW_2_mae(W_1, b_1, W_2, b_2, X, Y)

    W_1 -= dW_1 * learning_rate
    W_2 -= dW_2 * learning_rate

    b_1 -= db_1 * learning_rate
    b_2 -= db_2 * learning_rate

    print(f'Y / Y_prim: {list(zip(Y, Y_prim))}')
    print(f'loss: {loss}')

plt.plot(losses)
plt.title('loss')
plt.show()
