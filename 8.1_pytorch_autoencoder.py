import random

import torch.nn.functional as F

import scipy
import torch
import numpy as np
import matplotlib
import torchvision
from tqdm import tqdm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)

import torch.utils.data
import scipy.misc
import scipy.ndimage

model_path = 'D:\\project\\Autoencoder_weights.pt'
USE_PRETRAINED = True
USE_CUDA = torch.cuda.is_available()
MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
# if USE_CUDA:
#     MAX_LEN = None


class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.data = torchvision.datasets.EMNIST(
            root='D:/pythonProject/data',
            train=is_train,
            split='bymerge',
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def normalize(self, data):
        data_max = data.max()
        data_min = data.min()
        if data_max != data_min:
            data = ((data - data_min) / (data_max - data_min))
        return data

    def __getitem__(self, idx):
        pil_x, label_idx = self.data[idx]
        np_x = np.array(pil_x) # (28, 28)
        np_y = np.array(np_x)
        # make noise
        noise = np.random.rand(*np_x.shape)
        np_x = np.where(noise < 0.5, 0, np_y)

        np_x = np.expand_dims(np_x, axis=0) # (C, W, H)
        np_x = self.normalize(np_x)
        x = torch.FloatTensor(np_x)

        np_y = np.expand_dims(np_y, axis=0) # (C, W, H)
        np_y = self.normalize(np_y)
        y = torch.FloatTensor(np_y)

        np_label = np.zeros((len(self.data.classes),))
        np_label[label_idx] = 1.0
        label = torch.FloatTensor(np_label)
        return x, y, label


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetEMNIST(is_train=True),
    batch_size=16,
    shuffle=True,
    drop_last=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetEMNIST(is_train=False),
    batch_size=16,
    shuffle=False,
    drop_last=True
)

class LossCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim):
        return -torch.sum(y * torch.log(y_prim + 1e-20))


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=4,kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.Conv2d(in_channels=4,out_channels=8,kernel_size=4, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=32)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=5),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder.forward(x)
        out = self.decoder.forward(out)
        return out


model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if USE_PRETRAINED:
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # # load checkpoint
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # ..

if USE_CUDA and not USE_PRETRAINED:
    model = model.cuda()

if not USE_PRETRAINED:
    metrics = {}
    for stage in ['train', 'test']:
        for metric in [
            'loss'
        ]:
            metrics[f'{stage}_{metric}'] = []

    for epoch in range(1, 100):

        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}

            stage = 'train'
            if data_loader == data_loader_test:
                stage = 'test'

            for x, y, label in tqdm(data_loader):

                if USE_CUDA:
                    x = x.cuda()
                    y = y.cuda()

                y_prim = model.forward(x)
                loss = torch.mean((y_prim - y)**2)
                # TODO
                metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f

                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                loss = loss.cpu()
                y_prim = y_prim.cpu()
                x = x.cpu()
                y = y.cpu()

                np_y_prim = y_prim.data.numpy()
                idx_label = np.argmax(label, axis=1)

            metrics_strs = []
            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {round(value, 2)}')

            print(f'epoch: {epoch} {" ".join(metrics_strs)}')

        plt.clf()
        plt.subplot(121) # row col idx
        plts = []
        c = 0
        for key, value in metrics.items():
            value = scipy.ndimage.gaussian_filter1d(value, sigma=2)
            plts += plt.plot(value, f'C{c}', label=key)
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])

        for i, j in enumerate([4, 5, 6, 16, 17, 18]):
            plt.subplot(4, 6, j)
            plt.title(f"Class {data_loader.dataset.data.classes[idx_label[i]]}")
            plt.imshow(x[i][0].T, cmap=plt.get_cmap('Greys'))
            plt.subplot(4, 6, j+6)
            plt.imshow(np_y_prim[i][0].T, cmap=plt.get_cmap('Greys'))

        plt.tight_layout(pad=0.5)
        plt.draw()
        plt.pause(0.1)

        # save model weights
        torch.save(model.state_dict(), model_path)

        # # save chepoint
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss
        # }, model_path)
else:

    def normalize(data):
        data_max = data.max()
        data_min = data.min()
        if data_max != data_min:
            data = ((data - data_min) / (data_max - data_min))
        return data




    for x, y, label in data_loader_test:
        label = label.numpy()
        with torch.no_grad():
            embedding = model.encoder.forward(y)
            embedding_flat = embedding.view(y.size(0), -1)

        plt.clf()
        for idx, sample in enumerate(embedding_flat):
            value = 1 - F.cosine_similarity(embedding_flat[0], sample, 0)
            plt.subplot(2,1,1)
            plt.imshow(y[0][0].numpy().T, cmap=plt.get_cmap('Greys'))
            plt.title(f'Cos Distance {value}')
            plt.subplot(2,1,2)
            plt.imshow(y[idx][0].numpy().T, cmap=plt.get_cmap('Greys'))
            plt.tight_layout(pad=0.5)
            plt.draw()
            plt.pause(1)


input('quit?')
