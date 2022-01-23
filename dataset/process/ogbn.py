import numpy as np
import scipy.sparse as sp
import argparse
import os
from ogb.nodeproppred import PygNodePropPredDataset


def get_adj(edge_index):
    N = edge_index.shape[1]
    data = np.ones(N)
    row_ind = edge_index[0]
    col_ind = edge_index[1]
    adj = sp.csr_matrix((data, (row_ind, col_ind)), shape=(edge_index.max() + 1, edge_index.max() + 1))
    return adj


def process(graph, dataset):
    if dataset == "ogbn-mag":
        edge_index = graph.edge_index_dict[('paper', 'cites', 'paper')]
        adj = get_adj(edge_index)
        features = graph.x_dict['paper'].numpy()
        labels = graph.y_dict['paper'].numpy().ravel()
    else:
        assert False, "process dataset invalid"
    return adj, features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default='ogbn-mag', type=str)
    args = parser.parse_args()
    dataset = PygNodePropPredDataset(name=args.dataset)
    graph = dataset[0]
    adj, features, labels = process(graph, args.dataset)
    PATH = os.path.abspath(os.path.expanduser("~/GraphData/datasets/")) + os.sep
    filename = PATH + args.dataset + '.npz'
    np.savez(filename, adj_matrix=adj, node_attr=features, node_label=labels)
