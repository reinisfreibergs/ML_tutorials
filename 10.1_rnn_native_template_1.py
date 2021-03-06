import math
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from torch.hub import download_url_to_file
import torch.utils.data
import string
# pip install nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

import csv_result_parser as result_parser
from file_utils import FileUtils
FileUtils.createDir('10_results')

# nltk.download('punkt')

BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 1e-3

RNN_HIDDEN_SIZE = 256
RNN_LAYERS = 2
RNN_DROPOUT = 0.3

run_path = ''

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

MIN_SENTENCE_LEN = 3
MAX_SENTENCE_LEN = 20
MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
if DEVICE == 'cuda':
    MAX_LEN = 1000

PATH_DATA = './datasets'
os.makedirs('./results', exist_ok=True)
os.makedirs(PATH_DATA, exist_ok=True)


class DatasetCustom(torch.utils.data.Dataset):
    def __init__(self):
        if not os.path.exists(f'{PATH_DATA}/quotes.json'):
            download_url_to_file(
                'https://www.kaggle.com/akmittal/quotes-dataset/version/1?select=quotes.json',
                f'{PATH_DATA}/quotes.json',
                progress=True
            )
        with open(f'{PATH_DATA}/quotes.json', encoding='utf8') as fp:
            data_json = json.load(fp)

        self.sentences = []
        self.lengths = []
        self.words_to_idxes = {}
        self.words_counts = {}
        self.idxes_to_words = {}

        for each_instruction in data_json:
            str_instructions = each_instruction['Quote']

            exclist = string.punctuation + string.digits
            table = str.maketrans('', '', exclist)
            str_instructions_punctuation = str_instructions.translate(table)

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

        self.end_token = '[END]'
        self.words_to_idxes[self.end_token] = len(self.words_to_idxes)
        self.idxes_to_words[self.words_to_idxes[self.end_token]] = self.end_token
        self.words_counts[self.end_token] = len(self.sentences)

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
        # TODO placeholder to replace rare words
        # TODO remove punctuation
        # TODO histogtam of words_counts
        self.histogram_dict = dict( (k, v) for k, v in self.words_counts.items() if v <= 2  )
        rare_word_count = 0
        for k in self.histogram_dict.keys():
            print(k)
            rare_word_count+=1
        print(rare_word_count)
        # plt.bar(self.histogram_dict.keys() , self.histogram_dict.values(), width=3, color = 'g')
        # plt.show()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        np_x_idxes = np.array(self.sentences[idx] + [self.words_to_idxes[self.end_token]])
        np_x_padded = np.zeros((self.max_length, self.max_classes_tokens))
        np_x_padded[np.arange(len(np_x_idxes)), np_x_idxes] = 1.0

        np_y_padded = np.roll(np_x_padded, shift=-1, axis=0)
        np_length = self.lengths[idx]

        return np_x_padded, np_y_padded, np_length


torch.manual_seed(0)
dataset_full = DatasetCustom()
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

class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        stdv = 1 / math.sqrt(hidden_size)

        self.W_x = torch.nn.Parameter(
            torch.FloatTensor(input_size, hidden_size).uniform_(-stdv, stdv)
        )

        self.W_h = torch.nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size).uniform_(-stdv, stdv)
        )

        self.b = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).zero_()
        )

    def forward(self, x: PackedSequence, hidden=None):
        h_out = []

        x_unpacked, lengths = pad_packed_sequence(x, batch_first=True)
        batch_size = x_unpacked.size(0)
        if hidden is None:
            hidden = torch.FloatTensor(batch_size, self.hidden_size).zero_().to(DEVICE)

        x_seq = x_unpacked.permute(1, 0, 2)
        for x_t in x_seq:
            hidden = torch.tanh(
                (
                        self.W_x @ x_t.unsqueeze(dim=-1) +
                        self.W_h @ hidden.unsqueeze(dim=-1)
                ).squeeze() +
                self.b
            )
            h_out.append(hidden)
        t_h_out = torch.stack(h_out)
        t_h_out = t_h_out.permute(1, 0, 2)

        t_h_packed = pack_padded_sequence(t_h_out, lengths, batch_first=True)

        return t_h_packed

class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        stdv = 1 / math.sqrt(hidden_size)

        self.W_z = torch.nn.Parameter(
            torch.FloatTensor(input_size, hidden_size).uniform_(-stdv, stdv)
        )

        self.U_z = torch.nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size).uniform_(-stdv, stdv)
        )

        self.b_z = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).zero_()
        )
        self.W_r = torch.nn.Parameter(
            torch.FloatTensor(input_size, hidden_size).uniform_(-stdv, stdv)
        )

        self.U_r = torch.nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size).uniform_(-stdv, stdv)
        )

        self.b_r = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).zero_()
        )
        self.W_h = torch.nn.Parameter(
            torch.FloatTensor(input_size, hidden_size).uniform_(-stdv, stdv)
        )

        self.U_h = torch.nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size).uniform_(-stdv, stdv)
        )

        self.b_h = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).zero_()
        )

    def forward(self, x: PackedSequence, hidden=None):
        h_out = []

        x_unpacked, lengths = pad_packed_sequence(x, batch_first=True)
        batch_size = x_unpacked.size(0)
        if hidden is None:
            hidden = torch.FloatTensor(batch_size, self.hidden_size).zero_().to(DEVICE)


        x_seq = x_unpacked.permute(1, 0, 2)
        for x_t in x_seq:
            hidden = hidden.unsqueeze(dim=-1)
            x_t = x_t.unsqueeze(dim=-1)
            z_t = torch.sigmoid((self.W_z @ x_t).squeeze() + (self.U_z @ hidden).squeeze() + self.b_z)
            r_t = torch.sigmoid((self.W_r @ x_t).squeeze() + (self.U_r @ hidden).squeeze() + self.b_r)
            h_t = torch.tanh((self.W_h @ x_t).squeeze() + (self.U_h @ (r_t.unsqueeze(dim=-1) * hidden)).squeeze() + self.b_h)

            hidden = (1 - z_t) * hidden.squeeze() + z_t * h_t

            h_out.append(hidden)
        t_h_out = torch.stack(h_out)
        t_h_out = t_h_out.permute(1, 0, 2)

        t_h_packed = pack_padded_sequence(t_h_out, lengths, batch_first=True)

        return t_h_packed

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.embeddings = torch.nn.Embedding(
            num_embeddings=dataset_full.max_classes_tokens,
            embedding_dim=RNN_HIDDEN_SIZE
        )

        layers = []
        for _ in range(RNN_LAYERS):
            layers.append(RNNCell(
                input_size=RNN_HIDDEN_SIZE,
                hidden_size=RNN_HIDDEN_SIZE
            ))
        self.rnn  =torch.nn.Sequential(*layers)

    def forward(self, x: PackedSequence, hidden=None):
        x_idxes = x.data.argmax(dim=1)
        embs = self.embeddings.forward(x_idxes)
        embs_seq = PackedSequence(
            data=embs,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        hidden = self.rnn.forward(embs_seq)
        y_prim_logits = hidden.data @ self.embeddings.weight.t()
        y_prim = torch.softmax(y_prim_logits, dim=1)
        y_prim_packed = PackedSequence(
            data=y_prim,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        return y_prim_packed, hidden

model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

metrics = {}
best_test_loss = float('Inf')
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

filename = result_parser.run_file_name()
for epoch in range(1, EPOCHS+1):

    metrics_csv = []
    metrics_csv.append(epoch)
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y, lengths in data_loader:

            x = x.float().to(DEVICE)
            y = y.float().to(DEVICE)

            idxes = torch.argsort(lengths, descending=True)
            lengths = lengths[idxes]
            max_len = int(lengths.max())
            x = x[idxes, :max_len]
            y = y[idxes, :max_len]
            x_packed = pack_padded_sequence(x, lengths, batch_first=True)
            y_packed = pack_padded_sequence(y, lengths, batch_first=True)

            y_prim_packed, _ = model.forward(x_packed)

            weights = torch.from_numpy(dataset_full.weights[torch.argmax(y_packed.data, dim=1).cpu().numpy()])
            weights = weights.unsqueeze(dim=1).to(DEVICE)
            loss = -torch.mean(weights * y_packed.data * torch.log(y_prim_packed.data + 1e-8))

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
                metrics_strs.append(f'{key}: {round(value, 2)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if best_test_loss > loss.item() and epoch%49 ==0:
        best_test_loss = loss.item()
        torch.save(model.cpu().state_dict(), f'./results/model-{epoch}.pt')
        model = model.to(DEVICE)

    print('Examples:')
    y_prim_unpacked, lengths_unpacked = pad_packed_sequence(y_prim_packed.cpu(), batch_first=True)
    y_prim_unpacked = y_prim_unpacked[:5] # 5 examples
    for idx, each in enumerate(y_prim_unpacked):
        length = lengths_unpacked[idx]

        y_prim_idxes = np.argmax(each[:length].data.numpy(), axis=1).tolist()
        x_idxes = np.argmax(x[idx, :length].cpu().data.numpy(), axis=1).tolist()
        y_prim_idxes = [x_idxes[0]] + y_prim_idxes
        print('x     : ' +' '.join([dataset_full.idxes_to_words[it] for it in x_idxes]))
        print('y_prim: ' +' '.join([dataset_full.idxes_to_words[it] for it in y_prim_idxes]))
        print('')

    plt.figure(figsize=(12,5))
    plts = []
    c = 0
    for key, value in metrics.items():
        metrics_csv.append(value[-1])
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.show()
    if epoch%20==0:
        plt.savefig(f'./results/epoch-{epoch}.png')
    if epoch%299 == 0:
        plt.show()
        plt.savefig(f'./results/epoch-{epoch}.png')

    result_parser.run_csv(file_name='10_results/' + filename,
                      metrics=metrics_csv)

result_parser.best_result_csv(result_file='10.1_comparison_results.csv',
                            run_file='10_results/' + filename,
                            run_name=filename,
                            batch_size= BATCH_SIZE,
                            learning_rate= LEARNING_RATE)
