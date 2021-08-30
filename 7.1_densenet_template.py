import random
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
BATCH_SIZE = args.batch_size

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
            torch.nn.Linear(in_features=num_channels, out_features=158),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers.forward(x)

model = DenseNet()
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
            y = y.to(DEVICE).squeeze()

            y_prim = model.forward(x)

            indexes = range(len(y_prim))
            y_prim_out = y_prim[indexes, y]

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
    
    result_parser.run_csv(file_name=args.results_dir + filename,
                          metrics=metrics_csv)

result_parser.best_result_csv(result_file=args.comparison_file,
                                run_file=args.results_dir + filename,
                                run_name=filename,
                                batch_size= args.batch_size,
                                learning_rate=args.learning_rate)
