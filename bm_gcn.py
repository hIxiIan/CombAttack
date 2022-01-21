import sklearn.neural_network
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import random
import numpy as np
import os.path as osp
import os
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from copy import deepcopy
from utils import to_tensor, normalize_adj_tensor, is_sparse_tensor, accuracy
from graphgallery.datasets import NPZDataset
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score


def random_seed(seed=None):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BMGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None, seed=None):

        super(BMGCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass

        half_nhid = int(nhid / 2)
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, half_nhid, with_bias=with_bias)
        self.gc3 = GraphConvolution(half_nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.labels = None

    def forward(self, x, adj, softmax=True):
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)

        if self.with_relu:
            x = F.relu(self.gc2(x, adj))
        else:
            x = self.gc2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc3(x, adj)
        if softmax:
            x = F.log_softmax(x, dim=1)
        return x

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, args=None, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            adj, features, labels = to_tensor(adj=adj, features=features, labels=labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if is_sparse_tensor(adj):
                adj_norm = normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            if args is None:
                self._train_without_val(labels, idx_train, train_iters, verbose)
            else:
                self._train_with_args(labels, args, train_iters, verbose)

    def _train_with_args(self, labels, args, train_iters, verbose):
        phishing_train_id = args.phishing_train_id
        unlabeled_train_ids = args.unlabeled_train_ids
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for t in range(args.T):
            idx_train = phishing_train_id + unlabeled_train_ids[t]

            for epoch in range(train_iters):
                optimizer.zero_grad()
                output = self.forward(self.features, self.adj_norm, softmax=True)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                loss_train.backward()
                optimizer.step()

            if verbose and t % 10 == 0:
                print('Epoch {}, training loss: {}'.format(t, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm, softmax=True)
        self.output = output

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def test(self, idx_test):
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, adj=None, softmax=True):
        self.eval()
        if adj is None:
            return self.forward(self.features, self.adj_norm, softmax=softmax)

        if type(adj) is not torch.Tensor:
            adj = to_tensor(adj=adj, device=self.device)

        if is_sparse_tensor(adj):
            adj_norm = normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = normalize_adj_tensor(adj)

        return self.forward(self.features, adj_norm, softmax=softmax)


class BMMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None, seed=None):
        super(BMMLP, self).__init__()

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.seed = seed

        self.m1 = nn.Linear(nfeat, nhid, bias=with_bias)
        self.m2 = nn.Linear(nhid, nclass, bias=with_bias)

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.features = None
        self.labels = None

    def forward(self, x, softmax=True):
        if self.with_relu:
            x = F.relu(self.m1(x))
        else:
            x = self.m1(x)

        output = self.m2(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def initialize(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def fit(self, features, labels, adj=None, idx_train=None, args=None, train_iters=200, initialize=True, verbose=False, normalize=True):
        if initialize:
            self.initialize()

        if type(features) is not torch.Tensor or type(labels) is not torch.Tensor:
            features, labels = to_tensor(features=features, labels=labels, device=self.device)
        else:
            features = features.to(self.device)
            labels = labels.to(self.device)
        self.features = features
        self.labels = labels

        if args is not None:
            self._train_with_args(args, labels, train_iters, verbose)
        else:
            self._train_without_val(labels, idx_train, train_iters, verbose)

    def _train_with_args(self, args, labels, train_iters, verbose):
        phishing_train_id = args.phishing_train_id
        unlabeled_train_ids = args.unlabeled_train_ids

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for t in range(args.T):
            idx_train = phishing_train_id + unlabeled_train_ids[t]

            for epoch in range(train_iters):
                optimizer.zero_grad()
                output = self.forward(self.features, softmax=True)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                loss_train.backward()
                optimizer.step()

            if verbose and t % 10 == 0:
                print('Epoch {}, training loss: {}'.format(t, loss_train.item()))

        self.eval()
        output = self.forward(self.features, softmax=True)
        self.output = output

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, softmax=True)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, softmax=True)
        self.output = output

    @torch.no_grad()
    def predict(self, features=None, softmax=True):
        self.eval()
        if features is None:
            return self.forward(self.features, softmax=softmax)

        if type(features) is not torch.Tensor:
            features = to_tensor(features=features, device=self.device)

        return self.forward(features, softmax=softmax)

    def get_acc(self, idx_test=None):
        if idx_test is None:
            idx_test = self.idx_test
        acc = accuracy(self.output[idx_test], self.labels[idx_test]).item()
        return acc


def get_unlabeled_train_id(args, graph, train=0.2, test=0.8):
    features = graph.node_attr
    labels = graph.node_label
    phishing_train_id = list(np.where(labels == 1)[0])
    unlabeled_id = list(np.where(labels == 0)[0])

    X_train, X_test, y_train, y_test = train_test_split(phishing_train_id, labels[phishing_train_id], train_size=train, random_state=args.seed)
    unlabeled_train_ids = []
    phishing_nums = len(X_train)
    assert len(unlabeled_id) >= phishing_nums, "get_unlabeled_train_set invalid"

    for i in range(args.T):
        cur_unlabeled_id = random.sample(unlabeled_id, phishing_nums)
        unlabeled_train_ids.append(cur_unlabeled_id)

    args.phishing_train_id = X_train
    args.unlabeled_train_ids = unlabeled_train_ids
    args.idx_test = X_test

    return features, labels


#todo: cuda下bmgcn相同随机种子参数下，模型结果不一样
def get_bmgcn(args, graph, adjs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features, labels = get_unlabeled_train_id(args, graph, train=args.train_size)

    combined_embeddings = None
    gcns = []
    for i, adj in enumerate(adjs):
        gcn = BMGCN(nfeat=features.shape[1], nhid=16, nclass=labels.max() + 1, device=device)
        gcn = gcn.to(device)
        gcn.fit(features, adj, labels, None, args=args, train_iters=20, verbose=False)
        output = gcn.predict(adj=gcn.adj_norm, softmax=False)
        if i == 0:
            combined_embeddings = output
        else:
            combined_embeddings = torch.cat((combined_embeddings, output), 1)
        gcns.append(gcn)

    bmmlp = BMMLP(nfeat=combined_embeddings.shape[1], nhid=16, nclass=labels.max() + 1, device=device)
    bmmlp.idx_test = args.idx_test
    bmmlp = bmmlp.to(device)
    bmmlp.fit(combined_embeddings, labels, args=args, train_iters=20, verbose=False)
    bmmlp.output = bmmlp.output.detach().cpu()
    bmmlp.gcns = gcns
    return bmmlp


def get_bmgcn_sur(args):
    root = osp.abspath(osp.expanduser("~/GraphData/datasets/"))
    bmbc = np.load(''.join([root, os.sep, "bm" + args.dataset + ".npz"]), allow_pickle=True)
    A = bmbc["A"].item()
    V = bmbc["V"].item()
    F = bmbc["F"].item()
    bmmlp = get_bmgcn(args, args.graph, [A, V, F])
    logits = bmmlp.output.numpy()
    output = np.asarray([logit.argmax() for logit in logits])
    return output


def get_output(bmmlp, adjs, args, is_eva=True):
    if is_eva:
        combined_embeddings = None
        for i, gcn in enumerate(bmmlp.gcns):
            gcn.eval()
            adj = to_tensor(adj=adjs[i], device=bmmlp.device)
            if is_sparse_tensor(adj):
                adj_norm = normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = normalize_adj_tensor(adj)
            output = gcn.predict(adj=adj_norm, softmax=False)
            if i == 0:
                combined_embeddings = output
            else:
                combined_embeddings = torch.cat((combined_embeddings, output), 1)
        bmmlp.eval()
        output = bmmlp.predict(combined_embeddings).cpu().numpy()
    else:
        tmp_bmmlp = get_bmgcn(args, args.graph, adjs)
        output = tmp_bmmlp.output.numpy()

    return output


# X_train, X_test, y_train, y_test = train_test_split(combined_embeddings, labels,
#                                                     train_size=args.train_size, random_state=args.seed)
# model = MLPClassifier(hidden_layer_sizes=(16), activation="relu", random_state=args.seed)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# cr = classification_report(y_pred, y_test)
# auc = roc_auc_score(y_pred, y_test)
# print(cr)
# print(auc)

# X_train, X_test, y_train, y_test = train_test_split([i for i in range(len(combined_embeddings))], labels, stratify=labels,
#                                                     train_size=args.train_size, random_state=args.seed)
# args.idx_test = X_test
# args.idx_test = np.array(X_test)[np.array(y_test) == 1]
# model.fit(combined_embeddings, labels, idx_train=X_train, train_iters=200, verbose=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='bc1', help='dataset')
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument("--train_size", default=0.5, type=float)
    args = parser.parse_args()
    data = NPZDataset(args.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")
    graph = data.graph
    random_seed(args.seed)

    bmbc = np.load(''.join([data.root, os.sep, "bm" + args.dataset + ".npz"]), allow_pickle=True)
    adj_A = bmbc["A"].item()
    adj_V = bmbc["V"].item()
    adj_F = bmbc["F"].item()
    bmgcn = get_bmgcn(args, graph, [adj_A, adj_V, adj_F])
    acc = bmgcn.get_acc()
    print("acc:{}".format(acc))



