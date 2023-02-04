import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import numpy as np


class SignedLayer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(SignedLayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.W1 = nn.Linear(in_dim, in_dim)
        self.W2 = nn.Linear(in_dim, in_dim)
        nn.init.xavier_normal_(self.W1.weight, gain=1.414)
        nn.init.xavier_normal_(self.W2.weight, gain=1.414)

    def edge_applying(self, edges):
        e_real = edges.dst['d'] * edges.src['d'] * self.g.edata['w_real']
        e_real = self.dropout(e_real)
        e_imag = edges.dst['d'] * edges.src['d'] * self.g.edata['w_imag']
        e_imag = self.dropout(e_imag)
        return {'e_real': e_real, 'e_imag': e_imag}

    def forward(self, h_real, h_imag):
        self.g.ndata['h_real'] = h_real
        

        self.g.ndata['h_imag'] = h_imag
        self.g.apply_edges(self.edge_applying)

        self.g.update_all(fn.u_mul_e('h_real', 'e_real', 'm_real1') , fn.sum('m_real1', 'z_real1'))
        self.g.update_all(fn.u_mul_e('h_imag', 'e_imag', 'm_real2') , fn.sum('m_real2', 'z_real2'))
        self.g.ndata['z_real'] = self.g.ndata['z_real1'] - self.g.ndata['z_real2']

        self.g.update_all(fn.u_mul_e('h_real', 'e_imag', 'm_imag1') , fn.sum('m_imag1', 'z_imag1'))
        self.g.update_all(fn.u_mul_e('h_imag', 'e_real', 'm_imag2') , fn.sum('m_imag2', 'z_imag2'))
        self.g.ndata['z_imag'] = self.g.ndata['z_imag1'] + self.g.ndata['z_imag2']

        self.g.ndata['z_real'] = self.W1(self.g.ndata['z_real']) - self.W2(self.g.ndata['z_imag'])
        self.g.ndata['z_imag'] = self.W2(self.g.ndata['z_real']) + self.W1(self.g.ndata['z_imag'])
        return self.g.ndata['z_real'], self.g.ndata['z_imag']


class SignedMagnet(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(SignedMagnet, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(SignedLayer(self.g, hidden_dim, dropout))

        self.t1_real = nn.Linear(in_dim, hidden_dim)
        self.t1_imag = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim * 2, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1_imag.weight, gain=1.414)
        nn.init.xavier_normal_(self.t1_imag.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_real = torch.relu(self.t1_real(h))
        h_imag = torch.relu(self.t1_imag(h))
        # h = F.dropout(h, p=self.dropout, training=self.training)
        # raw = h
        for i in range(self.layer_num):
            h_real, h_imag = self.layers[i](h_real, h_imag)
            h_real = torch.relu(h_real)
            h_imag = torch.relu(h_imag)
            # h = self.eps * raw + h
        h = torch.cat([h_real, h_imag], dim=1)
        h = self.t2(h)
        return F.log_softmax(h, 1)


