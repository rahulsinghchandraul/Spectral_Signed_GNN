import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import random
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data import *


def preprocess_data(dataset, train_ratio):

    if dataset in ['wikielection']:

        edge = np.loadtxt('data/WikiElection.txt'.format(dataset), dtype=int).tolist()
        labels = np.loadtxt('data/WikiElection_labels.txt'.format(dataset), dtype=int)


        n = len(labels.tolist())
        idx = [i for i in range(n)]
        random.shuffle(idx)
        r0 = int(n * train_ratio)
        r1 = int(n * 0.1)
        idx_train = np.array(idx[:r0])
        idx_val = np.array(idx[r0:r1])
        idx_test = np.array(idx[r1:])

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))

        g.edata['w'] = torch.FloatTensor([e[2] for e in edge])
        features = np.diag(np.ones(n))
        features = torch.FloatTensor(features)

        nclass = 2
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(idx_train)
        val = torch.LongTensor(idx_val)
        test = torch.LongTensor(idx_test)
        print(dataset, nclass)
        
        return g, nclass, features, labels, train, test, val


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


