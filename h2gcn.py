import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
import torch.optim as optim
from torch import FloatTensor
import scipy.sparse as sp
from graphgallery.datasets import NPZDataset
import graphgallery.functional as gf
import graphgallery as gg


def norm_feat(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    row_sum = np.array(features.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    r_inv = np.power(row_sum.astype(np.float), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    return features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sp_to_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not isinstance(sparse_mx, sp.coo_matrix):
        sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.LongTensor([sparse_mx.row, sparse_mx.col])
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape
    )


class H2GCN(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, class_dim: int, k: int = 2, dropout: float = 0.5):
        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.k = k
        self.w_embed = nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        device = csp.values().device
        zero = torch.zeros(csp.values().shape[0]).to(device)
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, csp.values(), zero),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        device = sp1.device
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        ).to(device)

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1).to(device)
        self.a2 = self._adj_norm(a2).to(device)

    def forward(self, adj: torch.sparse.Tensor, x: FloatTensor) -> FloatTensor:
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [F.relu(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(torch.cat([r1, r2], dim=1))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        return torch.softmax(torch.mm(r_final, self.w_classify), dim=1)

    def evaluate(self, idx=None, verbose=0):
        if idx is None:
            idx = self.test
        acc = test(self, self.checkpoint_path, self.adj, self.features, self.labels, idx)[1]

        class results:
            def __init__(self, acc):
                self.accuracy = acc
        return results(acc)

    def fit(self, ph1, ph2, verbose, epochs):
        return

    @torch.no_grad()
    def predict(self, idx=None, transform="softmax"):
        self.eval()
        return self.forward(self.adj, self.features)[idx].cpu().numpy()


def train(model, optimizer, adj, features, labels, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(adj, features)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate(model, adj, features, labels, idx_val):
    model.eval()
    with torch.no_grad():
        output = model(adj, features)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        return loss_val.item(), acc_val.item()


def test(model, checkpoint_path, adj, features, labels, idx_test):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    with torch.no_grad():
        output = model(adj, features)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return loss_test.item(), acc_test.item()


def main(model, patience, checkpoint_path, epochs, optimizer, adj, features, labels, idx_train, idx_test, idx_val, verbose=0):
    # begin_time = time.time()
    tolerate = 0
    best_loss = 1000
    acc = 0
    for epoch in range(epochs):
        loss_train, acc_train = train(model, optimizer, adj, features, labels, idx_train)
        loss_validate, acc_validate = validate(model, adj, features, labels, idx_val)
        if verbose > 0 and (epoch + 1) % 1 == 0:
            print(
                'Epoch {:03d}'.format(epoch + 1),
                '|| train',
                'loss : {:.3f}'.format(loss_train),
                ', accuracy : {:.2f}%'.format(acc_train * 100),
                '|| val',
                'loss : {:.3f}'.format(loss_validate),
                ', accuracy : {:.2f}%'.format(acc_validate * 100)
            )
        if loss_validate < best_loss:
            best_loss = loss_validate
            acc = acc_validate
            torch.save(model.state_dict(), checkpoint_path)
            tolerate = 0
        else:
            tolerate += 1
        if tolerate == patience:
            break
    # print("Train cost : {:.2f}s".format(time.time() - begin_time))
    acc = test(model, checkpoint_path, adj, features, labels, idx_test)[1]
    # print("Test accuracy : {}".format(acc))
    return acc


def preprocess(args, graph, device):
    features = norm_feat(args.node_attr)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(args.node_label)
    train_mask = np.zeros(len(args.node_label)).astype('bool')
    test_mask = np.zeros(len(args.node_label)).astype('bool')
    val_mask = np.zeros(len(args.node_label)).astype('bool')
    train_mask[args.splits.train_nodes] = True
    test_mask[args.splits.test_nodes] = True
    val_mask[args.splits.val_nodes] = True
    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(test_mask)
    val_mask = torch.BoolTensor(val_mask)
    feat_dim = features.shape[1]
    class_dim = len(np.unique(args.node_label))
    adj = graph.adj_matrix.tocoo()
    adj = sp_to_tensor(adj)
    return adj.to(device), features.to(device), labels.to(device), train_mask.to(device), test_mask.to(device), val_mask.to(device), feat_dim, class_dim


def get_H2GCN(args, graph, layer_nums=2):
    patience = 100
    dropout = 0
    hidden = 64
    wd = 5e-4
    k = layer_nums # number of embedding rounds
    lr = 0.01
    epochs = 200
    root = os.path.split(__file__)[0]
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    adj, features, labels, idx_train, idx_test, idx_val, feat_dim, class_dim = preprocess(args, graph, device)
    checkpoint_path = root + '/ckpt/%s.pt' % args.dataset
    if not os.path.exists(root + '/ckpt'):
        os.makedirs(root + '/ckpt')
    model = H2GCN(feat_dim=feat_dim, hidden_dim=hidden, class_dim=class_dim, k=k, dropout=dropout).to(device)
    model.adj = adj
    model.features = features
    model.test = idx_test
    model.checkpoint_path = checkpoint_path
    model.labels = labels
    optimizer = optim.Adam([{'params': model.params, 'weight_decay': wd}], lr=lr)
    acc = main(model, patience, checkpoint_path, epochs, optimizer, adj, features, labels, idx_train, idx_test, idx_val)
    return model, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-tn", "--target_nums", default=50, type=int, help="target nums")
    parser.add_argument("--dataset", default="cora", type=str, help="dataset")
    args = parser.parse_args()
    args.verbose = 0
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
    FH2GCN, acc = get_H2GCN(args, graph, layer_nums=2)