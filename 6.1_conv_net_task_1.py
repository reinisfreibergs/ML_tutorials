import sklearn.datasets
import torch
import numpy as np
import matplotlib
import torchvision
import torch.nn.functional
import matplotlib.pyplot as plt
import torch.utils.data
from scipy.ndimage import gaussian_filter1d
from csv import writer
import time

header = ['ConvNet1']

with open('6.4_ConvNet_comparison.csv', 'a',newline='') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(header)
    # f_object.close()

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

        y = torch.LongTensor([y_idx])

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


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=5, stride=2, padding=1,
                in_channels=1,
                out_channels=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                kernel_size=3, stride=1, padding=1,
                in_channels=2,
                out_channels=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                kernel_size=3, stride=1, padding=1,
                in_channels=4,
                out_channels=8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            )
        )
        self.fc = torch.nn.Linear(
            in_features=8*15*11,
            out_features=5749
        )

    def forward(self, x):
        z = self.layers.forward(x)
        z_reshaped = z.view(x.size(0), -1)

        y_logits = self.fc.forward(z_reshaped)
        y_prim = torch.softmax(y_logits, dim=1)

        return y_prim


model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

start = time.time()
for epoch in range(1, 30):
    plt.clf()
    metrics_csv = []
    metrics_csv.append(epoch)
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in data_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE).squeeze()
            y_prim = model.forward(x)


            indexes = range(len(y))
            y_prim_out = y_prim[indexes, y]
            loss = torch.sum(-1 * torch.log(y_prim_out + 1e-8))
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

            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.mean((np_y == idx_y_prim) * 1.0)
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
        metrics_csv.append(value[-1])
        value = gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    # plt.show()
    plt.draw()
    plt.pause(0.1)

    with open('6.4_ConvNet_comparison.csv', 'a',newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(metrics_csv)

dt =time.time() - start
with open('6.4_ConvNet_comparison.csv', 'a',newline='') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow([dt])

print(dt)
