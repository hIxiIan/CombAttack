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


def random_seed(seed=None):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

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
    """ 2 Layer Graph Convolutional Network.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCN.
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None):

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

    def forward(self, x, adj):
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
        return F.log_softmax(x, dim=1)

    # def forward(self, x, adj):
    #     if self.with_relu:
    #         x = F.relu(self.gc1(x, adj))
    #     else:
    #         x = self.gc1(x, adj)
    #
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = self.gc2(x, adj)
    #     return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, args=None, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = to_tensor(features, adj, labels, device=self.device)
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
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_with_args(self, labels, args, train_iters, verbose):
        phishing_train_id = args.phishing_train_id
        unlabeled_train_ids = args.unlabeled_train_ids
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for t in range(args.T):
            idx_train = phishing_train_id + unlabeled_train_ids[t]

            for epoch in range(train_iters):
                optimizer.zero_grad()
                output = self.forward(self.features, self.adj_norm)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                loss_train.backward()
                optimizer.step()

            if verbose and t % 10 == 0:
                print('Epoch {}, training loss: {}'.format(t, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
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

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
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
    def predict(self, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if adj is None:
            return self.forward(self.features, self.adj_norm)

        if type(adj) is not torch.Tensor:
            _, adj = to_tensor(self.features, adj, device=self.device)

        if is_sparse_tensor(adj):
            adj_norm = normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = normalize_adj_tensor(adj)

        return self.forward(self.features, adj_norm)

    def fit(self, ph1, ph2, verbose, epochs):
        return


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


def get_output(models, adjs=None):
    output = None
    for i, model in enumerate(models):
        model.eval()
        adj = None
        if adjs is not None:
            adj = adjs[i]
        if output is None:
            output = model.predict(adj=adj).detach().cpu()
        else:
            output = torch.cat((output, model.predict(adj=adj).detach().cpu()), 1)
    return F.log_softmax(output, dim=1)


def get_acc(args, models):
    output = get_output(models)
    labels = models[0].labels
    idx_test = args.idx_test
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    if args.verbose > 0:
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


def get_bmgcn(args, graph, adjs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features, labels = get_unlabeled_train_id(args, graph, train=args.train_size)
    models = []
    for adj in adjs:
        model = BMGCN(nfeat=features.shape[1], nhid=16, nclass=labels.max() + 1, device=device)
        model = model.to(device)
        model.fit(features, adj, labels, None, args=args, train_iters=20, verbose=False)
        models.append(model)

    get_acc(args, models)
    return models


def get_bmgcn_sur(args):
    root = osp.abspath(osp.expanduser("~/GraphData/datasets/"))
    bmbc = np.load(''.join([root, os.sep, "bm" + args.dataset + ".npz"]), allow_pickle=True)
    A = bmbc["A"].item()
    V = bmbc["V"].item()
    F = bmbc["F"].item()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features, labels = get_unlabeled_train_id(args, args.graph, train=args.train_size)
    models = []
    for adj in [A, V, F]:
        model = BMGCN(nfeat=features.shape[1], nhid=16, nclass=labels.max() + 1, device=device)
        model = model.to(device)
        model.fit(features, adj, labels, None, args=args, train_iters=20, verbose=False)
        models.append(model)
    output = get_output(models)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
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
    get_bmgcn(args, graph, [adj_A, adj_V, adj_F])



