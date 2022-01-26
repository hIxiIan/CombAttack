import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import random
import graphgallery as gg
import argparse
import graphgallery.functional as gf
import scipy.sparse as sp
import dgl

from graphgallery.datasets import NPZDataset
from args import ARGS
from dgl import DGLGraph
from dgl import function as fn


class FALayer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(FALayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


class FAGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h, sf=False):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        if sf:
            return F.softmax(h, 1)
        return F.log_softmax(h, 1)

    def evaluate(self, idx=None, verbose=0):
        if idx is None:
            idx = self.test
        self.eval()
        logp = self.forward(self.features)

        class results:
            def __init__(self, labels):
                self.accuracy = accuracy(logp[idx], labels[idx])
        return results(self.labels)

    def fit(self, ph1, ph2, verbose, epochs):
        return

    @torch.no_grad()
    def predict(self, idx=None, transform="softmax"):
        self.eval()
        sf = False
        if transform == "softmax":
            sf = True
        return self.forward(self.features, sf)[idx].cpu().numpy()


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def preprocess(args, graph, device):
    features = normalize_features(args.node_attr)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(graph.node_label)
    train = torch.LongTensor(args.splits.train_nodes)
    test = torch.LongTensor(args.splits.test_nodes)
    val = torch.LongTensor(args.splits.val_nodes)
    g = DGLGraph(graph.adj_matrix)
    g = dgl.to_simple(g)
    g = dgl.to_bidirected(g)
    g = dgl.remove_self_loop(g)

    return features.to(device), labels.to(device), train.to(device), test.to(device), val.to(device), g.to(device)


def get_FAGCN(args, graph):
    dataset = args.dataset
    lr = 0.01
    weight_decay = 5e-5
    epochs = 200
    patience = 100
    hidden = 32
    dropout = 0.5
    eps = 0.3
    layer_num = 2
    nclass = len(set(graph.node_label))
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    features, labels, train, test, val, g = preprocess(args, graph, device)

    if device == "cuda":
        deg = g.in_degrees().cuda().float().clamp(min=1)
    else:
        deg = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g.ndata['d'] = norm

    net = FAGCN(g, features.size()[1], hidden, nclass, dropout, eps, layer_num)
    net.features = features
    net.test = test
    net.labels = labels
    if device == "cuda":
        net.cuda()

    # create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # main loop
    dur = []
    los = []
    loc = []
    counter = 0
    min_loss = 100.0
    max_acc = 0.0

    for epoch in range(epochs):
        if epoch >= 3:
            t0 = time.time()

        net.train()
        logp = net(features)

        cla_loss = F.nll_loss(logp[train], labels[train])
        loss = cla_loss
        train_acc = accuracy(logp[train], labels[train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        logp = net(features)
        test_acc = accuracy(logp[test], labels[test])
        loss_val = F.nll_loss(logp[val], labels[val]).item()
        val_acc = accuracy(logp[val], labels[val])
        los.append([epoch, loss_val, val_acc, test_acc])

        if loss_val < min_loss and max_acc < val_acc:
            min_loss = loss_val
            max_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience and dataset in ['cora', 'citeseer', 'pubmed']:
            if args.verbose > 0:
                print('early stop')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        if args.verbose > 0:
            print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
            epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))

    if dataset in ['cora', 'citeseer', 'pubmed'] or 'syn' in dataset:
        los.sort(key=lambda x: x[1])
        acc = los[0][-1]
    else:
        los.sort(key=lambda x: -x[2])
        acc = los[0][-1]
    return net, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-tn", "--target_nums", default=50, type=int, help="target nums")
    parser.add_argument("--dataset", default="cora", type=str, help="dataset")
    args = parser.parse_args()
    gg.set_backend("th")

    data = NPZDataset(args.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")
    graph = data.graph
    gf.random_seed(args.seed, gg.backend())
    splits = data.split_nodes(random_state=15)
    targets = random.sample(list(splits.test_nodes), args.target_nums)
    args.splits = splits
    args.node_attr = graph.node_attr
    args.node_label = graph.node_label
    args.adj_matrix = graph.adj_matrix
    FAGCN, acc = get_FAGCN(args)