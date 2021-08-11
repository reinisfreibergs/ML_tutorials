import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional
import sklearn.datasets

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
MAX_LEN = 200
INPUT_SIZE = 64
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0

class DatasetOlivetti(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.Y = sklearn.datasets.fetch_olivetti_faces(
            return_X_y=True,
            download_if_missing=True
        )
        self.data = list(zip(self.X, self.Y))

    def __len__(self):
        # if MAX_LEN:
        #     return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        x_idx, y_idx = self.data[idx]

        np_x = np.array(x_idx)
        np_x_reshaped = np.reshape(np_x, (1,64,64))
        x = torch.FloatTensor(np_x_reshaped)

        y = np.zeros((40,))
        y[y_idx] = 1.0
        y = torch.FloatTensor(y)

        return x, y

init_dataset = DatasetOlivetti()
lengths = [int(len(init_dataset)*0.8), int(len(init_dataset)*0.2)]
subsetA, subsetB = torch.utils.data.random_split(init_dataset, lengths, generator=torch.Generator().manual_seed(0))

data_loader_train = torch.utils.data.DataLoader(
    dataset=subsetA,
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=subsetB,
    batch_size=BATCH_SIZE,
    shuffle=False
)

def get_out_size(in_size, padding, kernel_size, stride):
    return int((in_size + 2 * padding - kernel_size) / stride + 1)

class Conv2d(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.K = torch.nn.Parameter(
            torch.FloatTensor(kernel_size, kernel_size, in_channels, out_channels)
        )
        torch.nn.init.xavier_uniform_(self.K)

    def forward(self, x):
        batch_size = x.size(0)
        in_size = x.size(-1) #last dim from (B,C,W,H)
        out_size = get_out_size(in_size, self.padding, self.kernel_size, self.stride)

        out = torch.zeros(batch_size, self.out_channels, out_size, out_size).to(DEVICE)


        x_padded_size = in_size + self.padding *2
        x_padded = torch.zeros(batch_size, self.in_channels, x_padded_size, x_padded_size).to(DEVICE)
        x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x

        K = self.K.view(-1, self.out_channels) # -1 means everything else multiplied together -> self.kernel_size*self.kernel_size*self.out_channels

        i_out = 0
        for i in range(0,x_padded_size - self.kernel_size, self.stride):
            j_out = 0
            for j in range(0,x_padded_size - self.kernel_size, self.stride):
                x_part = x_padded[:,:, i:i+self.kernel_size, j:j+self.kernel_size]
                x_part = x_part.reshape(batch_size, -1) # self.kernel_size*self.kernel_size*self.out_channels

                out_part = x_part @ K # (B, out_channels)
                out[:, :, i_out, j_out] = out_part

                j_out +=1
            i_out += 1

        return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        out_channels = 12
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=6, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )

        o_1 = get_out_size(INPUT_SIZE, kernel_size=5, stride=2, padding=1)
        o_2 = get_out_size(o_1, kernel_size=3, stride=2, padding=1)
        o_3 = get_out_size(o_2, kernel_size=3, stride=2, padding=1)

        self.fc = torch.nn.Linear(
            in_features=out_channels * o_3 * o_3,
            out_features=40
        )

    def forward(self, x):
        batch_size = x.size(0) #x.size() = (B,C_in,W_in,H_in) -> (16(size), 1(grayscale), 28(size), 28(size))
        out = self.encoder.forward(x) #out.size() = (B, C_out, W_out, H_out) -> (16, 12, 0_3, 0_3)
        out_flat = out.view(batch_size, - 1) #out_flat.size() = (B, F) ->
        logits = self.fc.forward(out_flat) #y_prim.size() = (B, 10)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 150):
    images = []
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in data_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_prim = model.forward(x)
            loss = torch.mean(-y * torch.log(y_prim+ 1e-8))

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                if epoch % 50 == 0:
                    images.append(x)

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')


    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.show()


    if epoch % 50 == 0:
        k=1
        for image_batch in reversed(images):
            for picture in reversed(image_batch):

                np_picture = np.array(picture)
                np_picture_reshaped = np.reshape(np_picture, (64,64))

                imgplot = plt.imshow(np_picture_reshaped)
                if k < 17:
                    plt.legend('image',title = idx_y_prim[-k], facecolor = 'red', fontsize = 10, labelcolor = 'white')
                    print(idx_y_prim)
                    plt.show()
                    k+=1
