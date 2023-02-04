import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import numpy as np

class SpectralSGCN1Layer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(SpectralSGCN1Layer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(in_dim, in_dim)
        nn.init.xavier_normal_(self.W.weight, gain=1.414)

    def edge_applying(self, edges):
        e = edges.dst['d'] * edges.src['d'] * self.g.edata['w']
        e = self.dropout(e)
        return {'e': e}

    def forward(self, h):
        self.g.ndata['h'] = self.W(h)
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'z'))
        return self.g.ndata['z']


class SpectralSGCN1(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(SpectralSGCN1, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(SpectralSGCN1Layer(self.g, hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        raw = h
        for i in range(self.layer_num):
            h = torch.relu(self.layers[i](h))
            h = 1 * raw + h # can weight the self loops via "eps"
        h = self.t2(h)
        return F.log_softmax(h, 1)

class SpectralSGCN2Layer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(SpectralSGCN2Layer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        alpha = torch.tanh(self.gate(h2)).squeeze()
        e = alpha * edges.dst['d'] * edges.src['d'] * self.g.edata['w']
        e = self.dropout(e)
        return {'e': e, 'm': alpha}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'z'))

        return self.g.ndata['z']


class SpectralSGCN2(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(SpectralSGCN2, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(SpectralSGCN2Layer(self.g, hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = 0.2 * raw + h
        h = self.t2(h)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.bias = nn.Parameter(torch.Tensor(3))
        self.W2 = nn.Linear(h_feats, 3)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']