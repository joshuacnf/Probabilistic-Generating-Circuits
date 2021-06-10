# simple models

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import numpy.linalg as la
from tqdm import tqdm
import os
import pdb

# Simple model over n variables
# Enumerating over 2^n assignments
# Note: forward method does not return log probability
class Simple(nn.Module):
    def __init__(self, n, x=None):
        super().__init__()
        m = 1 << n

        self.n = n
        self.m = m

        W = None
        if x is None:
            W = torch.randn(m - 1)
        else:
            W = torch.zeros(m - 1)

            cnt = 0.0
            count = {}
            for mask in range(0, m):
                count[mask] = 0.0
            for i in range(0, x.shape[0]):
                mask = 0
                for j in range(0, n):
                    mask += x[i, j].item() << j
                count[mask] += 1.0
                if mask != 0:
                    cnt += 1.0
            for mask in range(1, m):
                count[mask] += 1.0 / (m - 1)

            for k in count:
                count[k] /= cnt + 1.0

            for mask in range(1, m):
                W[mask - 1] = torch.log(torch.Tensor([count[mask]]))

        self.W = nn.Parameter(W, requires_grad=True)

    def forward(self, x):
        n = self.n
        y = []
        for i in range(0, n):
            y.append(x[:, i] << i)
        y = torch.sum(torch.stack(y, -1), -1)

        w = nn.functional.softmax(self.W, dim=0)
        w = torch.cat((torch.zeros(1).to(x.device), w), -1)
        y = w[y]

        return y

class PGC(nn.Module):
    def __init__(self, n, partition, x=None):
        super().__init__()

        self.n = n
        self.partition = partition
        self.dpp_size = len(partition)

        dpp_size = self.dpp_size

        B = torch.randn(dpp_size, dpp_size)
        B_norm = torch.norm(B, dim=0)
        for i in range(0, dpp_size):
            B[:,i] /= B_norm[i]
        self.B = nn.Parameter(B, requires_grad=True)

        PCs = []
        for part in self.partition:
            m = len(part)
            if x is not None:
                PCs.append(Simple(m, x[:,part]))
            else:
                PCs.append(Simple(m))
        self.PCs = nn.ModuleList(PCs)

    def forward(self, x):
        n = self.n
        dpp_size = self.dpp_size
        batch_size = x.shape[0]

        p = []
        for i, part in enumerate(self.partition):
            p.append(self.PCs[i](x[:, part]))
        p = torch.stack(p, -1)

        eps = 1e-8
        I = torch.eye(dpp_size).to(x.device)
        L = torch.matmul(torch.transpose(self.B, 0, 1), self.B) + eps * I
        L0 = L.clone()
        L = L.unsqueeze(0).repeat(batch_size, 1, 1)
        L = L * p.unsqueeze(1)
        L[torch.diag_embed(1-p) == 1.0] = 1.0
        y = torch.logdet(L)        
        alpha = torch.logdet(L0 + I)
        return y - alpha

class LEnsemble(nn.Module):
    def __init__(self, n, B=None):
        super().__init__()
        self.n = n

        if B is None:
            B = torch.randn(n, n)
            B_norm = torch.norm(B, dim=0)
            for i in range(0, n):
                B[:,i] /= B_norm[i]
        self.B = nn.Parameter(B, requires_grad=True)

    def forward(self, x):
        n = self.n
        batch_size = x.shape[0]

        eps = 1e-8
        I = torch.eye(n).to(x.device)
        L = torch.matmul(torch.transpose(self.B, 0, 1), self.B) + eps * I
        L0 = L.clone()
        L = L.unsqueeze(0).repeat(batch_size, 1, 1)

        L[x == 0] = 0.0
        L[x.unsqueeze(1).repeat(1,n,1) == 0] = 0.0
        L[torch.diag_embed(1-x) == 1] = 1.0

        y = torch.logdet(L)
        return y - torch.logdet(L0 + I)

# a sum mixture over m models
# n variables
class Sum(nn.Module):
    def __init__(self, n, models):
        super().__init__()
        self.n = n
        self.m = len(models)

        W = torch.randn(self.m - 1)
        self.W = nn.Parameter(W, requires_grad=True)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        ys = []
        for model in self.models:
            ys.append(model(x))
        y = torch.stack(ys, 0)

        w = torch.cat((self.W, torch.zeros(1).to(x.device)), -1)
        w = nn.functional.softmax(w, dim=0)
        w = torch.log(w)

        y = w.unsqueeze(-1) + y
        y = torch.logsumexp(y, 0)
        if (y != y).any():
            print('Sum: ERROR!')
            pdb.set_trace()
            print('!')

        return y

# a product mixture over m models
# n variables
class Product(nn.Module):
    def __init__(self, n, models):
        super().__init__()
        self.n = n
        self.m = len(models)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        offset = 0
        ys = []
        for model in self.models:
            ys.append(model(x[:, offset:offset+model.n]))
            offset += model.n

        y = torch.stack(ys, 0)
        y = torch.sum(y, 0)

        return y