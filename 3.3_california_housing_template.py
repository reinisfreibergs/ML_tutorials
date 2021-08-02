import time
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import csv


# X, Y = sklearn.datasets.fetch_california_housing(return_X_y=True)
# Y.shape (N, )
# Y = np.expand_dims(Y, axis=1)

# X.shape (N, 8)
# Y.shape (N, 1)

# sample 1 => X[0, 0:8], Y[0, 0]
# sample 2 => X[1, 0:8], Y[1, 0]
# sample 3 => X[2, 0:8], Y[2, 0]

# TODO implement min max normalization for each dimension in X and Y
def normalization(data ,a , b):
    minimum = np.min(data, axis = 0)
    maximum = np.max(data, axis = 0)

    # return 2*((data - minimum)/(maximum-minimum) - 0.5 )
    return a*((data - minimum)/(maximum-minimum) - b )

# X = normalization(X, 2, 0.5)
# Y = normalization(Y, 2, 0.5)

class Variable(object):
    def __init__(self, value):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)


class LayerLinear(object):
    def __init__(self, in_features: int, out_features: int):
        self.W = Variable(
            # value=np.random.random((out_features, in_features))
            value=np.ones((out_features, in_features))
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


# linear_1 = LinearLayer(in_features=8, out_features=4)
# # relu
# linear_2 = LinearLayer(in_features=4, out_features=1)
# x = Variable(value=X[:10])
# out = linear_1.forward(x)
# # relu
# y_prim = linear_2.forward(out)
#
# # loss func
# # loss backward
# linear_2.backward()
# linear_1.backward()


class LayerReLU:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        #[1, -2, ...] x.value(>=0) -> [True, False ...]
        #[True, False] * 1 -> [1, 0]
        self.output = Variable(
            (x.value >= 0) * x.value
        )
        return self.output

    def backward(self):
        self.x.grad = (self.x.value >= 0) * self.output.grad

class LossMAE():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(y.value - y_prim.value))
        return loss

    def backward(self):
        self.y_prim.grad = (self.y_prim.value - self.y.value) / np.abs(self.y.value - self.y_prim.value)

class LossMSE():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean((y.value - y_prim.value)**2)
        return loss

    def backward(self):
        self.y_prim.grad = 2 * (self.y_prim.value - self.y.value)

class LossHuber():
    def __init__(self):
        self.y = None
        self.y_prim = None
        self.delta = 1

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = self.delta**2 * np.sum(np.sqrt(1 + (((y.value - y_prim.value)/self.delta)**2)) -1)
        return loss

    def backward(self):
        self.y_prim.grad = (self.y_prim.value - self.y.value)/ np.sqrt((self.y.value - self.y_prim.value)**2 + 1)


class Precision_std():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        # loss = np.mean((y.value - y_prim.value)**2)
        loss = (1/np.std(y.value)) * np.sqrt(np.mean((y.value - y_prim.value)**2))
        return loss

class Precision_max_min():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = (1 / (np.max(y.value) - np.min(y.value))) * np.sqrt(np.mean((y.value - y_prim.value)**2))
        return loss

class Precision_r_squared():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = 1 - (np.sum(((y_prim.value - y.value)**2)) / np.sum(((np.mean(y_prim.value) - y.value)**2)))
        return loss


class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=8, out_features=4),
            LayerReLU(),
            LayerLinear(in_features=4, out_features=4),
            LayerReLU(),
            LayerLinear(in_features=4, out_features=1)
        ]

    def forward(self, x: Variable):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
            if isinstance(layer, LayerLinear):
                variables.append(layer.W)
                variables.append(layer.b)
        return variables

class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= np.mean(param.grad, axis = 0) *self.learning_rate


LEARNING_RATE = 1e-4
BATCH_SIZE = 16

# model = Model()
# optimizer = OptimizerSGD(
#     model.parameters(),
#     LEARNING_RATE
# )
loss_fn = LossMAE()
# loss_fn = LossMSE()
# loss_fn = LossHuber()

precision_fn_1 = Precision_std()
precision_fn_2 = Precision_max_min()

np.random.seed(0)
converged_epoch = []
multipliers = [0.0001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 10]
# multipliers = [0.65, 0.7,0.8, 1, 1.2, 1.4, 10]
for a in multipliers:
    model = Model()
    optimizer = OptimizerSGD(
        model.parameters(),
        LEARNING_RATE
    )
    X, Y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    Y = np.expand_dims(Y, axis=1)
    X = normalization(X, a, a/2)
    Y = normalization(Y, a, a/2)

    idxes_rand = np.random.permutation(len(X))
    X = X[idxes_rand]
    Y = Y[idxes_rand]

    idx_split = int(len(X) * 0.8)
    dataset_train = (X[:idx_split], Y[:idx_split])
    dataset_test = (X[idx_split:], Y[idx_split:])

    np.random.seed(int(time.time()))

    losses_train = [] #for plotting
    losses_test = []
    precision_std = []
    precision_max_min = []
    for epoch in range(1, 1000000):

        for dataset in [dataset_train, dataset_test]:
            X, Y = dataset
            losses = []
            precision_std1 = []
            precision_max_min1 = []
            for idx in range(0, len(X)- BATCH_SIZE, BATCH_SIZE):
                x = X[idx:idx+BATCH_SIZE]
                y = Y[idx:idx+BATCH_SIZE]

                y_prim = model.forward(Variable(x))
                loss = loss_fn.forward(Variable(y), y_prim)
                precision1 = precision_fn_1.forward(Variable(y), y_prim)
                precision2 = precision_fn_2.forward(Variable(y), y_prim)

                losses.append(loss)
                precision_std1.append(precision1)
                precision_max_min1.append(precision2)

                if dataset == dataset_train:
                    loss_fn.backward()
                    model.backward()
                    optimizer.step()

            if dataset == dataset_train:
                losses_train.append(np.mean(losses))
            else:
                losses_test.append(np.mean(losses))
                precision_std.append(np.mean(precision_std1))
                precision_max_min.append(np.mean(precision_max_min1))

        # print(f'epoch: {epoch} losses_train: {losses_train[-1]} losses_test: {losses_test[-1]} precision_std_test: {precision_std[-1]} precision_max_min_test: {precision_max_min[-1]}')
        print(f'epoch: {epoch} losses_train: {losses_train[-1]} precision_std_test: {precision_std[-1]}')
        # if epoch % 1 == 0:
        #     plt.plot(losses_train)
        #     plt.plot(losses_test)
        #     plt.ion()
        #     plt.show()
        #     plt.pause(.001)
        # if epoch % 299 == 0:
        #     plt.plot(losses_train)
        #     plt.plot(losses_test)
        #     plt.show()

        if len(losses_train) > 1:
            if round(losses_train[-1], 6) == round(losses_train[-2], 6):
                converged_epoch.append([a, epoch, losses_train[-1], precision_std[-1]])
                break
        print(converged_epoch)

print(converged_epoch)
csv_file = open("4.4_convergence_test.csv", 'w')
csv_writer = csv.writer(csv_file, delimiter=",")
csv_writer.writerow(converged_epoch)

coefficient = [item[0] for item in converged_epoch]
epoch = [item[1] for item in converged_epoch]
plt.plot(coefficient, epoch)
plt.xlabel("coefficient a")
plt.ylabel("epochs till converges")
plt.show()
