import string
import os
import pdb
import pandas as pd
import seaborn as sn
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
import torch.utils.data
from tqdm import tqdm

# pip install nltk
# import nltk
# nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

import csv_result_parser as result_parser

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('-learning_rate', default=5e-3, type=float)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epochs', default=300, type=int)
parser.add_argument('-transformer_heads', default=8, type=int) #4
parser.add_argument('-transformer_layers', default=2, type=int) #8
parser.add_argument('-interval', default=10, type=int)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate

HIDDEN_SIZE = 64
TRANSFORMER_LAYERS = args.transformer_layers
DROPOUT = 0.1
run_name = 'add_full'

TRANSFORMER_HEADS = args.transformer_heads

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

MIN_SENTENCE_LEN = 3
MAX_SENTENCE_LEN = 20
MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
if DEVICE == 'cuda':
    MAX_LEN = None

PATH_DATA = './data'
os.makedirs('./results', exist_ok=True)
os.makedirs(PATH_DATA, exist_ok=True)

class DatasetCustom(torch.utils.data.Dataset):
    def __init__(self):

        with open(f'{PATH_DATA}/quotes_unique_2.csv', encoding='utf8') as fp:
            data_json = pd.read_csv(fp, engine='c')

        self.sentences = []
        self.lengths = []
        self.words_to_idxes = {}
        self.words_counts = {}
        self.idxes_to_words = {}

        for each_instruction in data_json['Name']:
            str_instructions = each_instruction

            exclist = string.punctuation + string.digits
            table_ = str.maketrans('', '', exclist)
            str_instructions_punctuation = each_instruction.translate(table_)
            sentences = sent_tokenize(str_instructions_punctuation)
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                if len(words) > MAX_SENTENCE_LEN:
                    words = words[:MAX_SENTENCE_LEN]
                if len(words) < MIN_SENTENCE_LEN:
                    continue
                sentence_tokens = []
                for word in words:
                    if word not in self.words_to_idxes:
                        self.words_to_idxes[word] = len(self.words_to_idxes)
                        self.idxes_to_words[self.words_to_idxes[word]] = word
                        self.words_counts[word] = 0
                    self.words_counts[word] += 1
                    sentence_tokens.append(self.words_to_idxes[word])
                self.sentences.append(sentence_tokens)
                self.lengths.append(len(sentence_tokens))
            if MAX_LEN is not None and len(self.sentences) > MAX_LEN:
                break

        self.max_length = np.max(self.lengths) + 1

        # self.end_token = '[END]'
        # self.words_to_idxes[self.end_token] = len(self.words_to_idxes)
        # self.idxes_to_words[self.words_to_idxes[self.end_token]] = self.end_token
        # self.words_counts[self.end_token] = len(self.sentences)

        self.max_classes_tokens = len(self.words_to_idxes)

        word_counts = np.array(list(self.words_counts.values()))
        self.weights = (1.0 / word_counts) * np.sum(word_counts) * 0.5

        print(f'self.sentences: {len(self.sentences)}')
        print(f'self.max_length: {self.max_length}')
        print(f'self.max_classes_tokens: {self.max_classes_tokens}')

        print('Example sentences:')
        samples = np.random.choice(self.sentences, 5)
        for each in samples:
            print(' '.join([self.idxes_to_words[it] for it in each]))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # np_x_idxes = np.array(self.sentences[idx] + [self.words_to_idxes[self.end_token]])
        np_x_idxes = np.array(self.sentences[idx])
        np_x_padded = np.zeros((self.max_length, self.max_classes_tokens))
        np_x_padded[np.arange(len(np_x_idxes)), np_x_idxes] = 1.0

        np_y_padded = np.roll(np_x_padded, shift=-1, axis=0)
        np_length = self.lengths[idx]

        return np_x_padded, np_y_padded, np_length


dataset_full = DatasetCustom()

torch.manual_seed(0)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full, lengths=[int(len(dataset_full)*0.8), len(dataset_full)-int(len(dataset_full)*0.8)])
torch.seed()

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        pe = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, idxes):
        return self.pe[idxes, :]


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.project_k = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.project_q = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.project_v = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        )

        self.norm_1 = torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)
        self.norm_2 = torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)

    def forward(self, x, lengths, atten):
        batch_size = x.size(0)
        seq_size = x.size(1)

        k = self.project_k.forward(x)
        q = self.project_q.forward(x)
        v = self.project_v.forward(x)

        k = k.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS)).transpose(1, 2)
        q = q.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS)).transpose(1, 2)
        v = v.view(batch_size, seq_size, TRANSFORMER_HEADS, int(HIDDEN_SIZE/TRANSFORMER_HEADS)).transpose(1, 2)

        atten_raw = q @ k.transpose(-1, -2) / np.sqrt(x.size(-1))

        mask = torch.tril(torch.ones(seq_size,seq_size)).to(DEVICE)
        atten_mask = atten_raw.masked_fill(mask==0, value = float('-inf'))
        for idx, length in enumerate(lengths):
            atten_mask[idx, :, length:] = float('-inf')
            atten_mask[idx, length:, :] = float('-inf')

        atten = torch.softmax(atten_mask, dim=-1)
        atten = atten.masked_fill(((atten>0) == False), value=0.0)
        out = atten @ v

        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_size, HIDDEN_SIZE)
        atten = atten.detach().mean(dim=1)

        out_1 = x + torch.dropout(out, p=DROPOUT, train=self.training)
        out_1_norm = self.norm_1.forward(out_1)

        out_2 = self.ff.forward(out_1_norm)
        out_3 = out_1_norm + out_2
        y_prim = self.norm_2.forward(out_3)

        return y_prim, lengths, atten

class TransformerLayer_2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = torch.nn.TransformerEncoderLayer(d_model=HIDDEN_SIZE,
                                                  nhead=TRANSFORMER_HEADS,
                                                  activation='relu',
                                                  dropout=DROPOUT,
                                                  batch_first=True)

        self.attention = torch.nn.MultiheadAttention(embed_dim = HIDDEN_SIZE,
                                                     num_heads= TRANSFORMER_HEADS,
                                                     dropout=DROPOUT,
                                                     batch_first=True)

    def forward(self, x, lengths, atten):
        seq_size = x.size(1)
        mask = torch.tril(torch.ones(seq_size,seq_size)).to(DEVICE)

        y_prim = self.layer.forward(x, mask)
        atten, weights = self.attention(x, x, x, attn_mask=mask)

        return y_prim, lengths, atten

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project_w_e = torch.nn.Embedding(
            num_embeddings=dataset_full.max_classes_tokens,
            embedding_dim=HIDDEN_SIZE
        )

        self.project_p_e = torch.nn.Embedding(
            num_embeddings=dataset_full.max_classes_tokens,
            embedding_dim=HIDDEN_SIZE
        )

        self.transformer = torch.nn.ModuleList(
            [TransformerLayer_2() for _ in range(TRANSFORMER_LAYERS)]
        )

        self.fc = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)

    def forward(self, x: PackedSequence):

        x_e = PackedSequence(
            data=self.project_w_e.forward(x.data.argmax(dim=1)),
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        x_e_unpacked, lengths = pad_packed_sequence(x_e, batch_first=True)

        pos_idxes = torch.arange(0, torch.max(lengths)).to(DEVICE)
        p_e = self.project_p_e.forward(pos_idxes)
        p_e = p_e.unsqueeze(dim=0)
        p_e = p_e.expand(x_e_unpacked.size())

        z = x_e_unpacked + p_e
        atten = None
        for layer in self.transformer:
            z, lengths, atten = layer.forward(z, lengths, atten)

        z_packed = pack_padded_sequence(z, lengths, batch_first=True)
        y_prim_logits = self.fc.forward(z_packed.data) @ self.project_w_e.weight.t()
        # y_prim = torch.softmax(y_prim_logits, dim=1)

        y_prim_packed = PackedSequence(
            data=y_prim_logits,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        return y_prim_packed, atten

model = Model()
model = model.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=None).to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

metrics = {}
best_test_loss = float('Inf')
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

filename = result_parser.run_file_name()
import time
time_average = 0
for epoch in range(1, EPOCHS+1):
    start = time.time()
    metrics_csv = []
    metrics_csv.append(epoch)
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y, lengths in tqdm(data_loader):

            x = x.float().to(DEVICE)
            y = y.float().to(DEVICE)

            idxes = torch.argsort(lengths, descending=True)
            lengths = lengths[idxes]
            max_len = int(lengths.max())
            x = x[idxes, :max_len]
            y = y[idxes, :max_len]
            if int(lengths.min()) == 0:
                pdb.set_trace()
            x_packed = pack_padded_sequence(x, lengths, batch_first=True)
            y_packed = pack_padded_sequence(y, lengths, batch_first=True)

            y_prim_packed, atten = model.forward(x_packed)

            loss = loss_fn.forward(y_prim_packed.data, torch.argmax(y_packed.data, dim=1))

            metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim_packed.data.cpu().data.numpy()
            np_y = y_packed.data.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 4)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')


        # imageio.imwrite(f'./results/{run_name}-epoch-{epoch}-atten-0.png', atten[0].cpu().data.numpy())
        # imageio.imwrite(f'./results/{run_name}-epoch-{epoch}-atten-l.png', atten[-1].cpu().data.numpy())
    if epoch % args.interval==0:
        print('Examples:')
        y_prim_unpacked, lengths_unpacked = pad_packed_sequence(y_prim_packed.cpu(), batch_first=True)
        y_prim_unpacked = y_prim_unpacked[:5]
        for idx, each in enumerate(y_prim_unpacked):
            length = lengths_unpacked[idx]

            y_prim_idxes = np.argmax(each[:length].data.numpy(), axis=1).tolist()
            x_idxes = np.argmax(x[idx, :length].cpu().data.numpy(), axis=1).tolist()
            y_prim_idxes = [x_idxes[0]] + y_prim_idxes
            print('x     : ' +' '.join([dataset_full.idxes_to_words[it] for it in x_idxes]))
            print('y_prim: ' +' '.join([dataset_full.idxes_to_words[it] for it in y_prim_idxes]))

            if epoch % (args.interval) ==0:
                attention_data = atten.cpu().detach().numpy()
                attention_matrix = attention_data[idx]

                frame = pd.DataFrame(attention_matrix).round(decimals=2)
                plt.figure(figsize = (13,10))
                words = [dataset_full.idxes_to_words[it] for it in x_idxes]
                heatmap = sn.heatmap(frame, annot=True, xticklabels = words, yticklabels = words)
                heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
                heatmap.xaxis.tick_top() # x axis on top
                heatmap.xaxis.set_label_position('top')
                plt.savefig(f'./results/attention_matrix-epoch-{epoch}-idx {idx}.png')


    plt.figure(figsize=(12,5))
    plts = []
    c = 0
    for key, value in metrics.items():
        metrics_csv.append(value[-1])
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    if epoch % (args.interval) ==0:
        plt.legend(plts, [it.get_label() for it in plts])
        plt.savefig(f'./results/{run_name}-epoch-{epoch}.png')

        torch.save(model.cpu().state_dict(), f'./results/{run_name}-model-{epoch}.pt')
        model = model.to(DEVICE)

    result_parser.run_csv(file_name='results/' + str(args.transformer_layers) + str(args.learning_rate) + str(args.transformer_heads) + filename,
                      metrics=metrics_csv)
    end = time.time() - start
    time_average += end
    print(time_average/(epoch))
result_parser.best_result_csv(result_file='11.1_comparison_results.csv',
                            run_file='results/' + str(args.transformer_layers) + str(args.learning_rate) + str(args.transformer_heads) + filename,
                            run_name=filename,
                            batch_size= BATCH_SIZE,
                            learning_rate= LEARNING_RATE,
                            transformer_heads=TRANSFORMER_HEADS,
                            transformer_layers=TRANSFORMER_LAYERS)
