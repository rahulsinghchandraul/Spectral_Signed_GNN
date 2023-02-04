import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph
from utils_magnet import accuracy, preprocess_data
from model_signed_magnet import SignedMagnet

# torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='wikielection')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=2e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
parser.add_argument('--train_ratio', type=float, default=0.01, help='Ratio of training set')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_sizes = [0.99, 0.98, 0.90, 0.50]

train_ratios = [0.01, 0.02, 0.05, 0.10]


for j in range(3):

    train_ratio = train_ratios[j]
    best_all = []

    for i in range(2):

        g, nclass, features, labels, train, test, val = preprocess_data(args.dataset, train_ratio)


        ### SVD features of dimention 64
        features = np.loadtxt('data/WikiElection_features.txt')
        features = torch.FloatTensor(features)

        # print(g)

        features = features.to(device)
        labels = labels.to(device)
        train = train.to(device)
        test = test.to(device)
        # val = val.to(device)

        g = g.to(device)
        deg = g.in_degrees().cuda().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm

        net = SignedMagnet(g, features.size()[1], args.hidden, nclass, args.dropout, args.eps, args.layer_num).cuda()


        # create optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)



        # main loop
        dur = []
        los = []
        loc = []
        # best_val_acc = 0
        best_test_acc = 0

        for epoch in range(args.epochs):
            if epoch >= 3:
                t0 = time.time()

            net.train()
            logp = net(features)

            cla_loss = F.nll_loss(logp[train], labels[train]) # change to  nll_loss for log_softmax
            loss = cla_loss
            train_acc = accuracy(logp[train], labels[train])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.eval()
            logp = net(features)
            # logp = net(g, features)
            test_acc = accuracy(logp[test], labels[test])
            # loss_val = F.nll_loss(logp[val], labels[val]).item()
            # val_acc = accuracy(logp[val], labels[val])
            # los.append([epoch, loss_val, val_acc, test_acc])
            los.append([epoch, test_acc])

            # if best_val_acc < val_acc:
            #     best_val_acc = val_acc

            if best_test_acc < test_acc:
                best_test_acc = test_acc



            if epoch >= 3:
                dur.append(time.time() - t0)

            # print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} (best {:.4f}) | Test {:.4f} (best {:.4f})| Time(s) {:.4f}".format(
            #     epoch, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc, np.mean(dur)))
            
            # print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Test {:.4f} (best {:.4f})| Time(s) {:.4f}".format(
            #     epoch, loss, train_acc, test_acc, best_test_acc, np.mean(dur)))

        best_all.append(best_test_acc)


    print("Train Ratio:", train_ratio)
    mean = np.mean(best_all)
    std = np.std(best_all)
    max = np.amax(best_all)

    print("Mean: {:.4f}  STD: {:.4f}  MAX: {:.4f}".format(mean, std, max))
