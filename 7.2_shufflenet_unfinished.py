import random
from functools import reduce
import sklearn.datasets
import torch
import numpy as np
import matplotlib
import torchvision
import torch.nn.functional
import matplotlib.pyplot as plt
import torch.utils.data
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
import argparse
from file_utils import FileUtils
import csv_result_parser as result_parser


parser = argparse.ArgumentParser()

parser.add_argument('-epochs', default=5, type=int)
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-results_dir', default='7_results/', type=str)
parser.add_argument('-comparison_file', default='7.5_comparison_results.csv', type=str)

args = parser.parse_args()
FileUtils.createDir(args.results_dir)


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
            download_if_missing=True,
            min_faces_per_person = 10 #157 persons, 4324 images
        )
        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0, stratify=self.Y)
        self.data = list(zip(self.X, self.Y))

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        # x_idx, y_idx = self.data[idx]

        np_x = np.array(idx[0])
        np_x_reshaped = np.reshape(np_x, (1,62,47))
        x = torch.FloatTensor(np_x_reshaped)

        y = torch.LongTensor([idx[1]])

        return x, y


init_dataset = Dataset_Lfw_people()
y = init_dataset.Y
subset_train, subset_test = train_test_split(init_dataset.data, test_size=0.2, random_state=0, stratify=y)

train_samples = torch.utils.data.SubsetRandomSampler(subset_train)
test_samples = torch.utils.data.SubsetRandomSampler(subset_test)

data_loader_train = torch.utils.data.DataLoader(
    dataset = init_dataset,
    batch_size=BATCH_SIZE,
    # shuffle=True,
    drop_last=True,
    sampler=train_samples
)

data_loader_test = torch.utils.data.DataLoader(
    dataset = init_dataset,
    batch_size=BATCH_SIZE,
    # shuffle=False,
    drop_last=True,
    sampler=test_samples
)


class Shuffle(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups,
            channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class ShuffleNetBlock(torch.nn.Module):
    def __init__(self, in_features, num_groups):
        super().__init__()

        # TODO implement layers
        # Shuffle is given above
        # DW convolution is Conv2D where in_features == groups
        # GConvolution id Conv2D with groups parameter
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_features,
                out_channels=in_features,
                groups=num_groups,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            torch.nn.BatchNorm2d(num_features=in_features),
            torch.nn.ReLU(),
            Shuffle(groups=num_groups),
            torch.nn.Conv2d(
                in_channels=in_features,
                out_channels=in_features,
                groups=in_features,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.BatchNorm2d(num_features=in_features),
            torch.nn.Conv2d(
                in_channels=in_features,
                out_channels=in_features,
                groups=num_groups,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            torch.nn.BatchNorm2d(num_features=in_features),
        )

    def forward(self, x):
        z = self.layers.forward(x)
        z_prim = z + x
        z_lower = torch.nn.ReLU().forward(z_prim)

        return z_lower


class Reshape(torch.nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
    def forward(self, x):
        return x.view(self.target_shape)

class ShuffleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 16
        num_groups = 4
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=num_channels,
                kernel_size=7,
                stride=1,
                padding=1
            ),
            ShuffleNetBlock(in_features=num_channels, num_groups=num_groups),
            ShuffleNetBlock(in_features=num_channels, num_groups=num_groups),
            ShuffleNetBlock(in_features=num_channels, num_groups=num_groups),
            ShuffleNetBlock(in_features=num_channels, num_groups=num_groups),
            torch.nn.AdaptiveAvgPool2d(output_size=1),  #(B, num_channels, 1, 1)
            Reshape(target_shape=(-1, num_channels)),
            torch.nn.Linear(in_features=num_channels, out_features=10),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.layers.forward(x)

# model = ShuffleNet()
# inp = torch.ones((BATCH_SIZE, 1, 28, 28))
# out = model.forward(inp)
model = ShuffleNet()
model = model.to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

filename = result_parser.run_file_name()
for epoch in range(1, args.epochs):
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
        metrics_csv.append(value[-1])
        value = gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    # plt.show()
    plt.draw()
    plt.pause(0.1)

    result_parser.run_csv(file_name=args.results_dir + filename,
                          metrics=metrics_csv)

result_parser.best_result_csv(result_file=args.comparison_file,
                                run_file=args.results_dir + filename,
                                batch_size= args.batch_size,
                                learning_rate=args.learning_rate)
