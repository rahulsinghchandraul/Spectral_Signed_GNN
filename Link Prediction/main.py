import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import networkx as nx
import dgl
from dgl import DGLGraph
from utils import compute_auc_binary, preprocess_data, compute_loss
from utils import structured_negative_sampling
from model_signed import SpectralSGCN1, SpectralSGCN2, MLPPredictor
# from sklearn.model_selection import train_test_split
# import scipy.sparse as sp


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='BitcoinAlpha')
parser.add_argument('--lr', type=float, default=0.025, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = 'cpu'

g, g_pos, g_neg, features, n, edge = preprocess_data(args.dataset)

g = dgl.add_self_loop(g)
features = features.to(device)
print(g, g_pos, g_neg)

###############################################

g = g.to(device)
# g = dgl.to_bidirected(g)
deg = g.in_degrees().float().clamp(min=1)
# deg = (g.in_degrees() + g.out_degrees()).float().clamp(min=1)
norm = torch.pow(deg, -0.5)
g.ndata['d'] = norm
g.ndata['feat'] = features

best_auc_all = []
for i in range(10):
    ###########################################
    # Split edge set for training and testing

    #positive edges
    u, v = g_pos.edges()

    eids = np.arange(g_pos.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.2)
    train_size = g_pos.number_of_edges() - test_size
    test_pos_g_pos_u, test_pos_g_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_g_pos_u, train_pos_g_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    size_train_g_pos_labels = train_pos_g_pos_u.shape[0]
    # print("Pos Training label numbers", size_train_g_pos_labels)

    size_test_g_pos_labels = test_pos_g_pos_u.shape[0]
    # print("Pos test label numbers", size_test_g_pos_labels)

    ## Remove test positively signed edges
    train_g_pos = dgl.remove_edges(g, eids[:test_size])

    #Negative edges 
    u, v = g_neg.edges()

    eids = np.arange(g_neg.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.2)
    train_size = g_neg.number_of_edges() - test_size
    test_pos_g_neg_u, test_pos_g_neg_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_g_neg_u, train_pos_g_neg_v = u[eids[test_size:]], v[eids[test_size:]]

    size_train_g_neg_labels = train_pos_g_neg_u.shape[0]

    size_test_g_neg_labels = test_pos_g_neg_u.shape[0]

    ## Remove test negative signed edges
    train_g = dgl.remove_edges(train_g_pos, eids[:test_size])

    # Find all negative edges and split them for training and testing
    u, v = g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = 2* (size_test_g_neg_labels + size_test_g_pos_labels)
    i_n , j_n , k_n = structured_negative_sampling(edge, n)

    neg_u = i_n.numpy()
    neg_v = k_n.numpy()

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # print(g, train_g_pos, train_g)# g: full graph, train_g_pos: g - test positive edges, train_g: train_g_pos - test negative edges

    train_pos_g = dgl.graph((torch.cat([train_pos_g_pos_u, train_pos_g_neg_u]), torch.cat([train_pos_g_pos_v, train_pos_g_neg_v])), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    # test_pos_g = dgl.graph((torch.cat([test_pos_g_pos_u, test_pos_g_neg_u]), torch.cat([test_pos_g_pos_v, test_pos_g_neg_v])), num_nodes=g.number_of_nodes())
    # test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    # Positive and negative edges for test, these are not negative sampling
    test_pos_g = dgl.graph((test_pos_g_pos_u, test_pos_g_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_pos_g_neg_u, test_pos_g_neg_v), num_nodes=g.number_of_nodes())

    ####---MODEL-------
 
    model = SpectralSGCN1(train_g, features.size()[1], args.hidden, args.hidden, args.dropout, args.eps, args.layer_num)
    # You can replace DotPredictor with MLPPredictor.
    pred = MLPPredictor(args.hidden)

    criterion = nn.CrossEntropyLoss()

    dgl.add_self_loop(train_g)
    # ----------- Optimizer -------------- #
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr = args.lr, weight_decay=args.weight_decay)

    # ----------- 4. training -------------------------------- #
    best_auc = 0
    for e in range(args.epochs):
        # forward
        h = model(train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score, size_train_g_pos_labels, size_train_g_neg_labels, criterion)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pos_score = F.softmax(pred(test_pos_g, h),  dim = 1)
            neg_score = F.softmax( pred(test_neg_g, h),  dim = 1)
            auc = compute_auc_binary(pos_score, neg_score, size_test_g_pos_labels, size_test_g_neg_labels)
            if best_auc < auc:
                best_auc = auc
            print('AUC', auc, "Best AUC:", best_auc)
    print("Best AUC:", best_auc)
    best_auc_all.append(best_auc)


print(best_auc_all)
mean = np.mean(best_auc_all)
std = np.std(best_auc_all)
print("Mean:{:.4f} Std: {:.4f}".format(mean,std))