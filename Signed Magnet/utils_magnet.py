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
from scipy import sparse
from sklearn.decomposition import TruncatedSVD



def preprocess_data(dataset, train_ratio):

    if dataset in ['wikielection']:

        labels = np.loadtxt('data/WikiElection_labels.txt'.format(dataset), dtype=int)

        dataset  = pd.read_csv("./data/wikielection_Aq125.csv").values#.tolist()
        dataset = dataset[:,0:4]
        edge = np.array(dataset)

        edges = dataset[:,0:2]

        edges = np.array(edges, dtype=np.int64).T

        edge_index = torch.tensor(edges, dtype=torch.long)

        n = labels.size

        idx = [i for i in range(n)]
        random.shuffle(idx)
        r0 = int(n * train_ratio)
        # r1 = int(n * (train_ratio + (1- train_ratio)/5))
        r1 = int(n * 0.1)
        idx_train = np.array(idx[:r0])
        idx_val = np.array(idx[r0:r1])
        idx_test = np.array(idx[r1:])

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))

        # g.edata['w'] = torch.FloatTensor([e[2] for e in edge])

        g.edata['w_real'] = torch.FloatTensor([e[2] for e in edge])
        g.edata['w_imag'] = torch.FloatTensor([e[3] for e in edge])

        # print(g.edata['w_real'])

        #normalization will make features degenerated
        #features = normalize_features(features)
        features = np.diag(np.ones(n))
        # features = np.random.rand(n,1000)
        # features = g.in_degrees().float().clamp(min=1)
        # features = np.repeat(features, 2)
        # features = np.reshape(features, (n,2))
        # print(features.shape)

        # features = create_spectral_features(pos_edges.tolist(), neg_edges.tolist(), n)

        features = torch.FloatTensor(features)

        nclass = 2
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(idx_train)
        val = torch.LongTensor(idx_val)
        test = torch.LongTensor(idx_test)
        # print(dataset, nclass)
        
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


def create_spectral_features(positive_edges, negative_edges, node_count):
    """
    Creating spectral node features using the train dataset edges.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    p_edges = positive_edges + [[edge[1], edge[0]] for edge in positive_edges]
    n_edges = negative_edges + [[edge[1], edge[0]] for edge in negative_edges]
    train_edges = p_edges + n_edges
    index_1 = [edge[0] for edge in train_edges]
    index_2 = [edge[1] for edge in train_edges]
    values = [1]*len(p_edges) + [-1]*len(n_edges)
    # shaping = (node_count + 4, node_count + 4)
    shaping = (node_count, node_count)
    signed_A = sparse.csr_matrix(sparse.coo_matrix((values, (index_1, index_2)),
                                                   shape=shaping,
                                                   dtype=np.float32))

    svd = TruncatedSVD(n_components=64,
                       n_iter=30,
                       random_state=14)
    svd.fit(signed_A)
    X = svd.components_.T
    return X