import scipy.sparse as sp
import numpy as np
import os
import torch
from deeprobust.graph.defense import RGCN, SimPGCN, GCN
from sklearn.model_selection import train_test_split

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit_.npz")
    return adj + adj.T, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


def accuracy(output, labels):
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def get_model(model_name, adj, features, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0])

    if model_name == "RobustGCN":
        attacked_model = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1,
                              nhid=32, lr=0.01, dropout=0, device=device)
    elif model_name == "SimPGCN":
        attacked_model = SimPGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1,
                                 nhid=64, lr=0.01, dropout=0, weight_decay=5e-4, device=device)
    elif model_name == "GCN":
        attacked_model = GCN(nfeat=features.shape[1], nhid=64, nclass=labels.max() + 1, device=device)
    else:
        assert False, "invalid deeprobust model:{}".format(model_name)
    attacked_model.to(device)
    attacked_model.fit(sp.csr_matrix(features), sp.csr_matrix(adj), labels, idx_train, idx_val, train_iters=200,
                       verbose=False)
    attacked_model.eval()
    output = attacked_model.output
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return attacked_model, acc_test.item()


if __name__ == "__main__":
    # 转换代码可能要改一些路径
    dataset_dir = 'reddit/'
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(dataset_dir)
    labels = np.array(list(y_train) + list(y_val) + list(y_test))
    nodes = list(train_index) + list(val_index) + list(test_index)
    adj = adj[nodes][:, nodes]
    features = features[nodes]
    PATH = os.path.abspath(os.path.expanduser("~/GraphData/datasets/")) + os.sep
    filename = PATH + 'reddit.npz'
    np.savez(filename, adj_matrix=adj, node_attr=features, node_label=labels)

    # run
    model_name = "GCN"
    model, acc = get_model(model_name, adj, features, labels)