import random

import torch
import numpy as np
import matplotlib
import torchvision
import torch.nn.functional
import matplotlib.pyplot as plt
import torch.utils.data
from scipy.ndimage import gaussian_filter1d


MAX_LEN = 200 # For debugging, reduce number of samples
BATCH_SIZE = 16

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0

class DatasetFassionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(
            root='../data',
            train=is_train,
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.array(pil_x)
        np_x = np.expand_dims(np_x, axis=0) # (1, W, H)

        x = torch.FloatTensor(np_x)
        np_y = np.zeros((10,))
        np_y[y_idx] = 1.0

        y = torch.FloatTensor(np_y)
        return x, y


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetFassionMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetFassionMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class DenseBlock(torch.nn.Module):
    def __init__(self, in_features, num_chains=4):
        super().__init__()

        self.chains = []

        for i in range(num_chains):
            out_features = (i+1)*in_features
            self.chains.append(torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.Conv2d(
                    in_channels=out_features,
                    out_channels=in_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ).to(DEVICE))

    def parameters(self):
        return reduce(lambda a, b: a+b, [list(it.parameters()) for it in self.chains])

    def to(self, device):
        for i in range(num_chains):
            self.chains[i] = self.chains[i].to(device)

    def forward(self, x):
        inp = x
        list_out = [x]
        for chain in self.chains:
            out = chain.forward(inp)
            list_out.append(out)
            inp = torch.cat(list_out, dim=1) #(batch, channel, H, W)

        return inp

class TransitionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            torch.nn.AvgPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )
    def forward(self, x):
        return self.layers.forward(x)

class Reshape(torch.nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(self.target_shape)

class DenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 16
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=num_channels,
                kernel_size=7,
                stride=1,
                padding=1
            ),
            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels+4*num_channels, out_features=num_channels),
            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels+4*num_channels, out_features=num_channels),
            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels+4*num_channels, out_features=num_channels),
            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels+4*num_channels, out_features=num_channels),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            Reshape(target_shape=(-1, num_channels)),
            torch.nn.Linear(in_features=num_channels, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers.forward(x)

model = DenseNet()
model = model.to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):
    plt.clf()

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in data_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_prim = model.forward(x)

            loss = torch.sum(-y*torch.log(y_prim + 1e-8))
            # Sum dependant on batch size => larger LR
            # Mean independant of batch size => smaller LR

            # y.to('cuda')
            # y.cuda()

            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.mean((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plts = []
    c = 0
    for key, value in metrics.items():
        value = gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.show()
