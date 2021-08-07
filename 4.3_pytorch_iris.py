import time
import torch
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-4
BATCH_SIZE = 5

def normalization(data):
    minimum = np.min(data, axis = 0)
    maximum = np.max(data, axis = 0)

    return 2*((data - minimum)/(maximum-minimum) - 0.5 )

X, Y = sklearn.datasets.load_iris(return_X_y=True)

np.random.seed(0)
idxes_rand = np.random.permutation(len(X))

X = normalization(X)

X = X[idxes_rand]
Y = Y[idxes_rand]


Y_idxes = Y
Y = np.zeros((len(Y), 3))
Y[np.arange(len(Y)), Y_idxes] = 1.0

idx_split = int(len(X) * 0.9)
dataset_train = (X[:idx_split], Y[:idx_split])
dataset_test = (X[idx_split:], Y[idx_split:])

np.random.seed(int(time.time()))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=16),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=16),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=3),
            torch.nn.Softmax()
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim

model = Model()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr = LEARNING_RATE
)


loss_plot_train = []
loss_plot_test = []
accuracy_plot_train = []
accuracy_plot_test = []

for epoch in range(1, 20000):

    for dataset in [dataset_train, dataset_test]:
        X, Y = dataset
        losses = []
        accuracys = []
        for idx in range(0, len(X)-BATCH_SIZE, BATCH_SIZE):
            x = X[idx:idx+BATCH_SIZE]
            y = Y[idx:idx+BATCH_SIZE]

            y_prim = model.forward(torch.FloatTensor(x))
            y = torch.FloatTensor(y)

            y_idx = torch.argmax(y, dim = 1, keepdim = True)
            y_prim_out = torch.gather(y_prim, 1, y_idx)
            idx_y_prim = torch.argmax(y_prim, dim=1)

            loss = torch.mean(y_prim * torch.log((y_prim + 1e-8) / (y + 1e-8))) - torch.sum(y_prim) + torch.sum(y)
            # loss = torch.mean(-torch.log(y_prim_out + 1e-8)) #LossCrossEntropy loss function

            losses.append(loss.item())
            # y = y.detach() #y.fn_grad
            # y_prim = y_prim.detach()
            # accuracy = torch.max(y_prim * y, dim=1)
            accuracy = torch.mean((y_idx == idx_y_prim) * 1.0)
            # accuracys.append(accuracy[0].tolist())
            accuracys.append(accuracy)

            if dataset == dataset_train:
                loss.backward()

                optimizer.step()

        if dataset == dataset_train:
            accuracy_plot_train.append((np.mean(accuracys)))
            loss_plot_train.append(np.mean(losses))
        else:
            accuracy_plot_test.append((np.mean(accuracys)))
            loss_plot_test.append(np.mean(losses))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]} accuracy_train: {accuracy_plot_train[-1]} accuracy_test: {accuracy_plot_test[-1]}')

    if epoch % 500 == 0:
        plt.subplot(2,1,1)
        plt.plot(loss_plot_train)
        plt.plot(loss_plot_test)
        plt.subplot(2,1,2)
        plt.plot(accuracy_plot_train)
        plt.plot(accuracy_plot_test)
        plt.show()
