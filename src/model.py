import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from random import shuffle
from data_preparation import group_by_users_price_with_id

device = torch.device("cpu")


class ReqNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, out_size, seq_len=5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.fc0 = nn.Linear(seq_len, hidden_size - 1)
        self.dp0 = nn.Dropout(0.5)
        self.act0 = nn.LeakyReLU()
        self.fc = nn.Linear(2 * hidden_size - 1, out_size)
        self.act = nn.LeakyReLU()

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, state

    def forward(self, x, hidden):
        x = torch.transpose(x, 0, 1)
        all_outputs, hidden = self.lstm(x, hidden)
        out = all_outputs[-1]  # We are interested only on the last output
        x = x.squeeze(2)
        x = torch.transpose(x, 0, 1)
        inc = self.fc0(x)
        inc = self.dp0(inc)
        inc = self.act0(inc)
        x = torch.cat((inc, out), 1)
        x = self.fc(x)
        x = self.act(x)
        return x, hidden

def get_data(_sessions = None, _products = None, _users = None):
    if _products is None or _sessions is None or _users is None:
        raw_data = pd.DataFrame(group_by_users_price_with_id())
    else:
        raw_data = pd.DataFrame(group_by_users_price_with_id(_sessions, _products, _users))
    raw_data = raw_data.to_numpy()
    min_value = raw_data.min()
    max_value = raw_data.max()
    test_data_seq = []
    test_data_targets = []
    users_ids = []
    for user in raw_data:
        users_ids.append(user[0])
        test_data_seq.append(torch.from_numpy(user[7:12]))
        test_data_targets.append(user[12])
    test_data_seq = (torch.stack(test_data_seq).float() - min_value) / max_value
    test_data_targets = (torch.Tensor(test_data_targets).float() - min_value) / max_value
    return test_data_seq, test_data_targets, users_ids


def get_trained_model(path):
    model = ReqNet(1, 3, 2, 1)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def process_data(model, data):
    with torch.no_grad():
        hidden, state = model.init_hidden(len(data))
        hidden, state = hidden.to(device), state.to(device)
        test_preds, _ = model(data.to(device).unsqueeze(2), (hidden, state))
    return test_preds.squeeze(1)


def check_accuracy(preds, targets):
    p_best = [int(i) for i in preds]
    t_best = [int(i) for i in targets]
    counter = 0
    for elem in p_best:
        if elem in t_best:
            counter+=1
    return counter/len(preds)


def choose_random_users(n_vouchers, data):
    _, _, ids = data
    p = deepcopy(ids)
    shuffle(p)
    p_best = []
    for v in range(n_vouchers):
        p_best.append(p[v])
    return p_best


def choose_best_users(n_vouchers, data):
    seq, _, ids = data
    model = get_trained_model(".//src//ReqNetTrained.pt")
    model.eval()
    preds = process_data(model, seq)
    p = deepcopy(preds.cpu().detach().numpy())
    p_best = []
    for i in range(n_vouchers):
        best_arg = np.argmax(p, 0)
        p_best.append(ids[best_arg])
        p[best_arg] = -1
    return p_best


def choose_true_best_users(n_vouchers, data):
    _, labels, ids = data
    p = deepcopy(labels.cpu().detach().numpy())
    p_best = []
    for i in range(n_vouchers):
        best_arg = np.argmax(p, 0)
        p_best.append(ids[best_arg])
        p[best_arg] = -1
    return p_best