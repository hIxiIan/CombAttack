import scipy.sparse as sp
import numpy as np
import os


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit_.npz")
    return adj + adj.T, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


if __name__ == "__main__":
    dataset_dir = 'reddit/'
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(dataset_dir)
    labels = np.array(list(y_train) + list(y_val) + list(y_test))
    nodes = list(train_index) + list(val_index) + list(test_index)
    adj = adj[nodes][:, nodes]
    features = features[nodes]
    PATH = os.path.abspath(os.path.expanduser("~/GraphData/datasets/")) + os.sep
    filename = PATH + 'reddit.npz'
    np.savez(filename, adj_matrix=adj, node_attr=features, node_label=labels)