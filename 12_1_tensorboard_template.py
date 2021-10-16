import time
import matplotlib
# import sklearn.datasets
from tensorboardX.utils import figure_to_image

# TODO make sure tensorboard_utils are found here depending on your project structure
import tensorboard_utils
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# pip install tensorflow
# pip install tensorboardX
# pip install argparse
# cmd: tensorboard --logdir=/abs/path/to/seq_default

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('-sequence_name', default=f'seq_default', type=str)
parser.add_argument('-is_cuda', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-learning_rate', default=3e-5, type=float)
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-epochs', default=10, type=int)
# TODO add more params
args = parser.parse_args()

#TODO add summary writer
summary_writer = tensorboard_utils.CustomSummaryWriter(
    logdir=f'{args.sequence_name}/{args.run_name}'
)

for epoch in range(1, args.epochs + 1):
    print(f'simulated epoch: {epoch}')

    train_loss = -np.log(epoch/(args.epochs + 2))
    acc = np.log(epoch)/np.log(args.epochs + 1)

    #TODO add to scalar
    summary_writer.add_scalar(
        tag='train_loss',
        scalar_value=train_loss,
        global_step=epoch
    )

    summary_writer.add_scalar(
    tag='train_acc',
    scalar_value=acc,
    global_step=epoch
    )

    summary_writer.add_hparams(
        hparam_dict=args.__dict__,
        metric_dict={
            'train_loss': train_loss,
            'train_acc': acc
        },
        global_step=epoch
    )

    class_count = 3
    conf_matrix = np.zeros((class_count, class_count))
    for batch in range(10):
        y = torch.softmax(torch.randn((args.batch_size, class_count)), dim=1)
        y_prim = torch.softmax(torch.randn((args.batch_size, class_count)), dim=1)

        y_idx = torch.argmax(y, dim=1).data.numpy()
        y_prim_idx = torch.argmax(y_prim, dim=1).data.numpy()
        for idx_sample in range(args.batch_size):
            conf_matrix[y_idx[idx_sample], y_prim_idx[idx_sample]] += 1

    fig = plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greys'))
    plt.xticks([0, 1, 2], ['Setosa', 'Virginica', 'ersicolor'])
    plt.yticks([0, 1, 2], ['Setosa', 'Virginica', 'ersicolor'])
    for x in range(class_count):
        for y in range(class_count):
            plt.annotate(
                str(round(100 * conf_matrix[x,y]/np.sum(conf_matrix[x]), 1)), xy=(y, x),
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='white'
            )
    plt.xlabel('True')
    plt.ylabel('Predicted')
    # plt.show()

    summary_writer.add_figure(
        tag='conf_matrix',
        figure=fig,
        global_step=epoch
    )

    # embeddings, classes = sklearn.datasets.make_blobs(n_samples=1000, n_features=128, centers=3)
    # summary_writer.add_embedding(
    #     mat=embeddings,
    #     metadata=classes.tolist(),
    #     tag='embeddings',
    #     global_step=epoch
    # )

    summary_writer.flush()
summary_writer.close()
