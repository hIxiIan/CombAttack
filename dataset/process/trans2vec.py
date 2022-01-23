import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import argparse
from CombAttack.tedge import tGraph, METHOD_MAP, load_labels


class ARGS():
    def __init__(self):
        self.tedge_type = "TBS"
        self.dimensions = 12
        self.num_walks = 1
        self.walk_length = 5
        self.window_size = 4
        self.workers = 8
        self.train_size = 0.5
        self.verbose = 0
        self.is_dan = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default='trans2vec', type=str)
    cmd = parser.parse_args()
    args = ARGS()
    args.dataset = cmd.dataset
    args.time_biased_type, args.first_biased_type, args.amount_biased, args.alpha = METHOD_MAP[args.tedge_type]
    tG = tGraph('../phishing/TransEdgelist.txt', verbose=1)

    # adj
    G = tG.G
    g = nx.DiGraph(G)
    adj = nx.to_scipy_sparse_matrix(g)
    adj.data = np.ones(len(adj.data))

    # labels
    slabels = load_labels('../phishing/label.txt') # 890
    N = adj.shape[0]
    labels = np.zeros(N).astype('int32')
    for k, v in slabels.items():
        if v == 1:
            labels[np.int32(k)] = 1

    # features
    amount_data = sp.lil_matrix((N, N), dtype=np.float64)
    timestamp_data = sp.lil_matrix((N, N), dtype=np.int64)
    # amount_timestamp_data = sp.lil_matrix((N, N), dtype=np.float64)
    # alpha = 0.5
    for i in range(N):
        cur = str(i)
        nbrs = list(G.neighbors(cur))
        for nbr in nbrs:
            latest_timestamp = list(G.get_edge_data(cur, nbr))
            amount = G[cur][nbr][latest_timestamp[-1]]['weight']
            amount_data[i, int(nbr)] = amount
            timestamp_data[i, int(nbr)] = len(latest_timestamp)
            # amount_timestamp_data[i, int(nbr)] = (amount**alpha) * (len(latest_timestamp)**(1-alpha))
    amount_data = amount_data.tocsr()
    timestamp_data = timestamp_data.tocsr()
    # amount_timestamp_data = amount_timestamp_data.tocsr()

    PATH = os.path.abspath(os.path.expanduser("~/GraphData/datasets/")) + os.sep
    filename = PATH + args.dataset + '.npz'
    np.savez(filename, adj_matrix=adj, amount_data=amount_data, timestamp_data=timestamp_data, node_label=labels)