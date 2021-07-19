import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

X, Y = sklearn.datasets.fetch_california_housing(return_X_y=True)
# Y.shape (N, )
Y = np.expand_dims(Y, axis=1)

# X.shape (N, 8)
# Y.shape (N, 1)

# sample 1 => X[0, 0:8], Y[0, 0]
# sample 2 => X[1, 0:8], Y[1, 0]
# sample 3 => X[2, 0:8], Y[2, 0]

# TODO implement min max normalization for each dimension in X and Y
print(X[:5])
print(Y[:5])


class Variable(object):
    def __init__(self, value):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)


class LinearLayer(object):
    def __init__(self, in_features: int, out_features: int):
        self.W = Variable(
            value=np.random.random((out_features, in_features))
        )
        self.b = Variable(
            value=np.zeros((out_features,))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            np.matmul(self.W.value, x.value.transpose()).transpose() + self.b.value
        )
        return self.output # this output will be input for next function in model

    def backward(self):
        # W*x + b / d b = 0 + b^{1-1} = 1
        # d_b = 1 * chain_rule_of_prev_d_func
        self.b.grad = 1 * self.output.grad

        # d_W = x * chain_rule_of_prev_d_func
        self.W.grad = np.matmul(
            np.expand_dims(self.output.grad, axis=2),
            np.expand_dims(self.x.grad, axis=1),
        )

        # d_x = W * chain_rule_of_prev_d_func
        self.x.grad = np.matmul(
            self.W.value.transpose(),
            self.output.grad.transpose()
        ).transpose()


linear_1 = LinearLayer(in_features=8, out_features=4)
# relu
linear_2 = LinearLayer(in_features=4, out_features=1)
x = Variable(value=X[:10])
out = linear_1.forward(x)
# relu
y_prim = linear_2.forward(out)

# loss func
# loss backward

linear_2.backward()
linear_1.backward()
