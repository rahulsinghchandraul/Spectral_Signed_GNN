import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import data
import torch
import random
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data import *

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)




def preprocess_data(dataset):

    if dataset in ['BitcoinAlpha']:

        dataset  = pd.read_csv("./Data/bitcoin_alpha.csv").values#.tolist()
        dataset = dataset[:,0:3]
        edge = np.array(dataset, dtype=np.int64)

        edges = dataset[:,0:2]

        pos_edges = np.asarray([edge[0:2] for edge in dataset if edge[2] > 0])
        neg_edges = np.asarray([edge[0:2] for edge in dataset if edge[2] < 0])

        edges = np.array(edges, dtype=np.int64).T

        edge_index = torch.tensor(edges, dtype=torch.long)

        n = np.amax(edge) + 1

        U = [e[0] for e in edge] 
        V = [e[1] for e in edge] 
        g = dgl.graph((U, V))
        # g = dgl.to_bidirected(g)

        g.edata['w'] = torch.FloatTensor([np.sign(e[2]) for e in edge])

        U = [e[0] for e in pos_edges] 
        V = [e[1] for e in pos_edges] 

        g_pos = dgl.graph((U,V))

        U = [e[0] for e in neg_edges] 
        V = [e[1] for e in neg_edges] 

        g_neg = dgl.graph((U,V))

        features = create_spectral_features(pos_edges.tolist(), neg_edges.tolist(), n)
        features = torch.FloatTensor(features)
        
        return g, g_pos, g_neg, features, n, edge_index


def structured_negative_sampling(edge_index, num_nodes):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.
    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (LongTensor, LongTensor, LongTensor)
    """

    i, j = edge_index#.to('cpu')
    idx_1 = i * num_nodes + j

    k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
    idx_2 = i * num_nodes + k

    mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
        idx_2 = i[rest] * num_nodes + tmp
        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        k[rest] = tmp
        rest = rest[mask.nonzero(as_tuple=False).view(-1)]

    return edge_index[0], edge_index[1], k.to(edge_index.device)

def compute_loss(pos_score, neg_score, size_pos, size_neg, criterion):
    scores = torch.cat([pos_score, neg_score])
    # print(scores.shape)
    labels = torch.cat([torch.zeros(size_pos) , torch.ones(size_neg), 2 * torch.ones(neg_score.shape[0])])
    labels = labels.long()
    # print(labels.shape)
    return criterion(scores, labels)



def compute_auc_binary(pos_score, neg_score, size_pos, size_neg):
    scores = torch.cat([pos_score, neg_score]).numpy()
    # scores = torch.exp(scores).numpy()
    predictions = scores[:,0]/ scores[:,0:2].sum(1)
    # print(predictions.shape)
    targets = torch.cat([torch.zeros(size_pos) , torch.ones(size_neg)]).numpy()
    targets = [0 if target == 1 else 1 for target in targets]
    # print(labels.shape)
    return roc_auc_score(targets, predictions)

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

    svd = TruncatedSVD(n_components=30,
                       n_iter=30,
                       random_state=14)
    svd.fit(signed_A)
    X = svd.components_.T
    return X


