import math
import time
import torch
import torchvision.datasets
import argparse
from torch.utils.data import DataLoader
import numpy as np
import json

#TODO replace path
import tensorboard_utils

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('-id', default=0, type=int)
parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('-sequence_name', default=f'seq_default', type=str)
parser.add_argument('-learning_rate', default=1e-2, type=float)
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-is_cuda', default=False, type=lambda x: (str(x).lower() == 'true'))
args = parser.parse_args()


if not torch.cuda.is_available() or not args.is_cuda:
    args.device = 'cpu'
    args.is_cuda = False
    print('cuda not available')
else:
    args.device = 'cuda'
    print(f'cuda devices: {torch.cuda.device_count()}')

#TODO initialize summary_writer of tensorboard_utils

MAX_LEN = 0 # for debugging on CPU, on testing set to 0

class DatasetFassionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(
            root='../data', #TODO data location for your dir structure
            train=is_train,
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        pil_x, y_idx = self.data[idx]
        np_x = np.array(pil_x)
        np_x = np.expand_dims(np_x, axis=0)
        x = torch.FloatTensor(np_x)
        np_y = np.zeros((10,))
        np_y[y_idx] = 1.0
        y = torch.FloatTensor(np_y)
        return x, y

data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetFassionMNIST(is_train=True),
    batch_size=args.batch_size,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetFassionMNIST(is_train=False),
    batch_size=args.batch_size,
    shuffle=False
)


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.resnet_pretrained = torchvision.models.resnet18(
            pretrained=True
        )

        conv1_weight = self.resnet_pretrained.conv1.weight
        self.resnet_pretrained.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=self.resnet_pretrained.conv1.out_channels,
            kernel_size=self.resnet_pretrained.conv1.kernel_size,
            padding=self.resnet_pretrained.conv1.padding,
            stride=self.resnet_pretrained.conv1.stride,
            bias=self.resnet_pretrained.conv1.bias
        )
        self.resnet_pretrained.conv1.weight.data = torch.mean(conv1_weight, dim=1, keepdim=True)
        self.resnet_pretrained.fc = torch.nn.Linear(
            in_features=self.resnet_pretrained.fc.in_features,
            out_features=10
        )

    def forward(self, x):
        return torch.softmax(self.resnet_pretrained.forward(x), dim=1)


model = Model(args).to(args.device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
)

metrics_best = {
}
metrics_best_coefs = {
}
modes = ['train', 'test']
for mode in modes:
    metrics_best[f'{mode}_loss_best'] = float('Inf')
    metrics_best_coefs[f'{mode}_loss_best'] = -1.0

    metrics_best[f'acc_best'] = float('-Inf')
    metrics_best_coefs['acc_best'] = 1.0

for epoch in range(1, args.epochs+1):
    print(f'epoch: {epoch}')

    metrics = {
    }
    for mode, dataloader in zip(modes, [data_loader_train, data_loader_test]):
        metrics[f'{mode}_loss'] = []

        if mode == 'train':
            model = model.train()
            torch.set_grad_enabled(True)
        else:
            model = model.eval()
            torch.set_grad_enabled(False)

        for x, y in dataloader:
            y_prim = model.forward(x.to(args.device))
            loss = torch.mean(-y.to(args.device) * torch.log(y_prim + 1e-20))

            metrics[f'{mode}_loss'].append(loss.to('cpu').item())

            #TODO calculate accuracy and store results
            #TODO [optional, extra task] store confusion matrix

            if mode == 'train':
                loss.backward()
                optimizer.step()

        for metric_name, metric in metrics.items():
            metrics[metric_name] = np.average(metrics[metric_name])
            if f'{metric_name}_best' in metrics_best:
                if metrics_best[f'{metric_name}_best'] * metrics_best_coefs[f'{metric_name}_best'] < metrics[metric_name] * metrics_best_coefs[f'{metric_name}_best']:
                    metrics_best[f'{metric_name}_best'] = metrics[metric_name]

    #TODO add to tensorboard scalar metrics
    #TODO [optional, extra task] add to tensorboard confusion matrix
    #TODO add tensorboard hparams

    print(json.dumps(metrics))
    print(json.dumps(metrics_best))

