import sklearn.datasets
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

class Dataset_Lfw_people(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.X, self.Y = sklearn.datasets.fetch_lfw_people(
            return_X_y=True,
            download_if_missing=True
        )
        self.data = list(zip(self.X, self.Y))

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        x_idx, y_idx = self.data[idx]

        np_x = np.array(x_idx)
        np_x_reshaped = np.reshape(np_x, (1,62,47))
        x = torch.FloatTensor(np_x_reshaped)

        y = np.zeros((5749,))
        y[y_idx] = 1.0
        y = torch.FloatTensor(y)

        return x, y


init_dataset = Dataset_Lfw_people()
lengths = [int(len(init_dataset)*0.8)+1, int(len(init_dataset)*0.2)]
subsetA, subsetB = torch.utils.data.random_split(init_dataset, lengths, generator=torch.Generator().manual_seed(0))


data_loader_train = torch.utils.data.DataLoader(
    dataset = subsetA,
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset = subsetB,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class ResBlock(torch.nn.Module):

    def __init__(self, in_features):
        super().__init__()

        self.upper_layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=3, stride=1, padding=1,
                in_channels=in_features,
                out_channels=in_features),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=in_features),
            torch.nn.Conv2d(
                kernel_size=3, stride=1, padding=1,
                in_channels=in_features,
                out_channels=in_features)
        )

        self.lower_layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=in_features)
        )

    def forward(self, x):
        z = self.upper_layers.forward(x)
        z_prim = z + x

        z_lower = self.lower_layers.forward(z_prim)

        return z_lower


class ResNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            ResBlock(in_features=1),
            torch.nn.Conv2d(
                kernel_size=3, stride=1, padding=1,
                in_channels=1,
                out_channels=32),
            torch.nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            ResBlock(in_features=32),
            torch.nn.Conv2d(
                kernel_size=3, stride=1, padding=1,
                in_channels=32,
                out_channels=64),
            torch.nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            ResBlock(in_features=64),
            torch.nn.Conv2d(
                kernel_size=3, stride=1, padding=1,
                in_channels=64,
                out_channels=128),
            torch.nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            ),
            torch.nn.AdaptiveAvgPool2d(output_size=(1,1))
        )
        self.fc = torch.nn.Linear(
            in_features=128,
            out_features=5749
        )

    def forward(self, x):
        z = self.layers.forward(x)
        z_reshaped = z.squeeze()
        y_logits = self.fc.forward(z_reshaped)
        y_prim = torch.softmax(y_logits, dim=1)

        return y_prim


model = ResNet()
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

            y_idx = torch.argmax(y, dim = 1)
            indexes = range(len(y_idx))
            y_prim_out = y_prim[indexes, y_idx]

            loss = torch.sum(-1*torch.log(y_prim_out + 1e-8))
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
    # plt.show()
    plt.draw()
    plt.pause(0.1)
