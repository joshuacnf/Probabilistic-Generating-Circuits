import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import numpy.linalg as la
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

import os
import math
import argparse

from models_simple import *

device = 'cuda'

class DatasetFromFile(Dataset):
    def __init__(self, filename):
        examples = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = [int(x) for x in line.split(',')]
                examples.append(line)
        x = torch.LongTensor(examples)
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)

def init():
    global device
    global CUDA_CORE

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2020)
    np.random.seed(2020)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', default='', type=str)
    arg_parser.add_argument('--dataset', default='', type=str)
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='0', type=str)
    arg_parser.add_argument('--model_path', default='', type=str)
    arg_parser.add_argument('--batch_size', default=8, type=int)
    arg_parser.add_argument('--output_file', default='', type=str)

    args = arg_parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args

def load_data(dataset_path, dataset,
            load_train=True, load_valid=True, load_test=True):
    dataset_path += '{}/'.format(dataset)
    train_path = dataset_path + '{}.train.data'.format(dataset)
    valid_path = dataset_path + '{}.valid.data'.format(dataset)
    test_path = dataset_path + '{}.test.data'.format(dataset)

    train, valid, test = None, None, None

    if load_train:
        train = DatasetFromFile(train_path)
    if load_valid:
        valid = DatasetFromFile(valid_path)
    if load_test:
        test = DatasetFromFile(test_path)

    return train, valid, test


def example_lls(model, dataset_loader):
    lls = []
    for x_batch in dataset_loader:
        x_batch = x_batch.to(device)
        y_batch = model(x_batch)
        lls_ = y_batch.tolist()
        lls += lls_
    return lls


def avg_ll(model, dataset_loader):
    lls = []
    dataset_len = 0
    for x_batch in dataset_loader:
        x_batch = x_batch.to(device)
        y_batch = model(x_batch)
        ll = torch.sum(y_batch)
        lls.append(ll.item())
        dataset_len += x_batch.shape[0]
    avg_ll = torch.sum(torch.Tensor(lls)).item() / dataset_len
    return avg_ll


def evaluate_model_on_data(model, data_loader, output_file):
    with open(output_file, 'w') as f:
        test_example_lls = example_lls(model, data_loader)
        for x in test_example_lls:
            f.write('{}\n'.format(x))


def evaluate_model(model, test,
                batch_size, output_file, dataset_name):
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)    

    model = model.to(device)
    model.eval()

    with open(output_file, 'w') as f:
        test_example_lls = example_lls(model, test_loader)
        for x in test_example_lls:
            f.write('{}\n'.format(x))


def main():
    args = init()

    _, _, test = load_data(args.dataset_path, args.dataset)

    print('test: {}'.format(test.x.shape))

    model = torch.load(args.model_path)

    evaluate_model(model, test,
        args.batch_size, args.output_file, args.dataset)

if __name__ == '__main__':
    main()