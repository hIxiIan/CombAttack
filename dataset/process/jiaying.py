import gc
import sys
import argparse
import os, pickle, random
import pandas as pd
import numpy as np
import networkx as nx
import logging
import warnings
import scipy.sparse as sp
from tqdm import tqdm
from sklearn import preprocessing
from time import time
from copy import deepcopy as dcopy

warnings.filterwarnings('ignore')
logging.basicConfig(filename="intesys.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)


def load_pickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def dump_pickle(data, fileName):
    with open(fileName, 'ab') as f:
        pickle.dump(data, f)


def random_walk(G, src, gsize=1000):
    node_set = set()
    node_set.add(src)
    last_step = src
    print("start walking ...")
    t1 = time()
    while len(node_set) < gsize:
        cur = last_step
        cur_nbs = list(nx.neighbors(G, cur))
        if len(cur_nbs) == 0:
            break
        if len(node_set) % 5000 == 0:
            print('|, cost:{} min'.format((time() - t1) / 60))
            sys.stdout.flush()
        next_node = random.choice(cur_nbs)
        node_set.add(next_node)
        last_step = next_node
    print('\nleghth of nodes set:', len(node_set))
    return list(node_set)


def sample_subgraph(mulG, muldG, src, gsize):
    t1 = time()
    src = src.lower()
    nodes_lis = random_walk(mulG, src, gsize)
    sp_mul_G = nx.subgraph(mulG, nodes_lis).copy()
    sp_mul_dG = nx.subgraph(muldG, nodes_lis).copy()
    sp_mul_G = nx.convert_node_labels_to_integers(sp_mul_G, first_label=0)
    sp_mul_dG = nx.convert_node_labels_to_integers(sp_mul_dG, first_label=0)
    print('sample_subgraph cost:{} min'.format((time() - t1) / 60))
    return nodes_lis, sp_mul_G, sp_mul_dG


def get_edges_features(mul_G, mul_dG):
    f_path = os.path.join(DATA_PATH, 'features.dat')
    e_path = os.path.join(DATA_PATH, 'edges.dat')
    we_path = os.path.join(DATA_PATH, 'weighted_edges.dat')
    print('Start writing edges ...')
    with open(e_path, 'w') as f:
        lines, weighted_edges = [], []
        for ind, edge in enumerate(nx.edges(mul_G)):
            (u, v) = edge
            eg = mul_G[u][v][0]
            amo, tim = eg['amount'], eg['timestamp']
            weighted_edges.append([u, v, amo, tim])
            lines.append(''.join([str(u), ' ', str(v)]))
        f.writelines('%s\n' % l for l in lines)

        df_wei = pd.DataFrame(weighted_edges, columns=['node1', 'node2', 'amount', 'timestamp'])
        norm_columns = ['amount', 'timestamp']
        df_wei[norm_columns] = preprocessing.minmax_scale(df_wei[norm_columns])
        df_wei['hybrid_fea'] = (df_wei['amount'] + df_wei['timestamp']) / 2
        df_wei = df_wei.drop(['amount', 'timestamp'], axis=1)
        df_wei.to_csv(we_path, sep=' ', index=False, header=False)
        print('Edges write done.')

    print('Start getting features ...')
    df_data = []
    for i, nd in tqdm(enumerate(mul_dG.nodes())):
        label = int(mul_dG.nodes[nd]['isp'])
        AF1 = mul_dG.in_degree[nd]
        AF2 = mul_dG.out_degree[nd]
        AF3 = AF1 + AF2
        AF4 = mul_dG.in_degree(nd, weight='amount')
        AF5 = mul_dG.out_degree(nd, weight='amount')
        AF6 = AF4 + AF5

        neighbors, timestamps = set(), []
        for ta, tb in mul_dG.in_edges(nd):
            neighbors.add(ta)
            timestamps.append(mul_dG[ta][tb][0]['timestamp'])
        for ta, tb in mul_dG.out_edges(nd):
            neighbors.add(tb)
            timestamps.append(mul_dG[ta][tb][0]['timestamp'])

        timestamps = list(map(float, timestamps))
        AF7 = len(neighbors)
        AF8 = (max(timestamps) - min(timestamps)) / AF3
        df_data.append([label, AF1, AF2, AF3, AF4, AF5, AF6, AF7, AF8])

    df = pd.DataFrame(df_data, columns=
    ['label', 'AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7', 'AF8'])
    df[['label']] = df[['label']].astype(int)
    norm_columns = ['AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7', 'AF8']
    df[norm_columns] = preprocessing.minmax_scale(df[norm_columns])
    dump_pickle(df, f_path)
    print('Features ready.')


def get_features_map(mul_dG, gsize):
    print('Start getting get_features_map ...')
    N = mul_dG.number_of_nodes()
    Aij = sp.lil_matrix((N, N), dtype=np.float64)
    Vij = sp.lil_matrix((N, N), dtype=np.float64)
    Fij = sp.lil_matrix((N, N), dtype=np.float64)
    for i, nd in enumerate(mul_dG.nodes()):
        for ta, tb in mul_dG.out_edges(nd):
            if Aij[ta, tb] == 0 and Vij[ta, tb] == 0:
                _d = list(mul_dG[ta][tb].values())
                _timestamp = [_dd['timestamp'] for _dd in _d]
                Vij[ta, tb] = np.var(_timestamp)
                Aij[ta, tb] = len(_d)
                if Aij[ta, tb] >= 2.:
                    Fij[ta, tb] = np.diff(_timestamp).sum() / Aij[ta, tb]
    Aij = Aij.tocsr()
    Vij = Vij.tocsr()
    Fij = Fij.tocsr()
    fname = int(gsize / 10000)
    np.savez('C://Users/pc/GraphData/datasets/bmbc' + str(fname) + '.npz', A=Aij, V=Vij, F=Fij)
    print('get_features_map ready.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sg", "--SAMPLE_GSIZE", default=50, type=int)
    args = parser.parse_args()

    PATH = os.path.abspath(os.path.expanduser("~/GraphData/datasets/")) + os.sep
    RAWDATA_PATH = PATH + 'rawdata/'
    PUBLICDATA_PATH = PATH + 'publicdata/'
    SAMPLE_GSIZE = args.SAMPLE_GSIZE
    SAMPLE_MULGS_PATH = os.path.join(PUBLICDATA_PATH, 'graph_%d/SP_MulGs.pkl' % SAMPLE_GSIZE)
    SAMPLE_MULDIGS_PATH = os.path.join(PUBLICDATA_PATH, 'graph_%d/SP_MulDiGs.pkl' % SAMPLE_GSIZE)
    DATA_PATH = os.path.join(PUBLICDATA_PATH, 'graph_%d' % SAMPLE_GSIZE)
    FEATURES_PATH = os.path.join(PUBLICDATA_PATH, 'graph_%d/features.dat' % SAMPLE_GSIZE)
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    print('', SAMPLE_MULGS_PATH, '\n', SAMPLE_MULDIGS_PATH, '\n', DATA_PATH)

    t1 = time()
    not_ccmulG = load_pickle(os.path.join(RAWDATA_PATH, 'MulGraph.pkl'))
    not_ccmuldG = load_pickle(os.path.join(RAWDATA_PATH, 'MulDiGraph.pkl'))
    largest_connected_components = list(max(nx.connected_components(not_ccmulG), key=len))
    const_ccmulG  = nx.subgraph(not_ccmulG, largest_connected_components)
    const_ccmuldG = nx.subgraph(not_ccmuldG, largest_connected_components)
    del not_ccmuldG, not_ccmulG
    gc.collect()
    ccmulG, ccmuldG = const_ccmulG, const_ccmuldG
    print('cost:{} min'.format((time() - t1)/ 60))
    print('const_ccmulG.number_of_nodes: ', const_ccmulG.number_of_nodes())

    src_act = random.choice(list(ccmulG.nodes()))
    invalid_cnt = 0

    nodes_lis, sp_mul_G, sp_mul_dG = sample_subgraph(ccmulG, ccmuldG, src_act, gsize=SAMPLE_GSIZE)
    dump_pickle(sp_mul_G, SAMPLE_MULGS_PATH)
    dump_pickle(sp_mul_dG, SAMPLE_MULDIGS_PATH)
    print(src_act)
    print('sampled graph connected:', nx.is_connected(sp_mul_G))
    print('sampled graph info:', '\n', nx.info(sp_mul_G))

    # sp_mul_G  = load_pickle(SAMPLE_MULGS_PATH)
    # sp_mul_dG = load_pickle(SAMPLE_MULDIGS_PATH)
    print(type(sp_mul_G))
    print(nx.edges(sp_mul_G, 1))
    print(nx.is_connected(sp_mul_G))
    print(nx.info(sp_mul_dG))
    get_edges_features(sp_mul_G, sp_mul_dG)
    get_features_map(sp_mul_dG, SAMPLE_GSIZE)

    # 转化成npz
    df = load_pickle(FEATURES_PATH)
    y_cols_name = ['label']
    x_cols_name = [x for x in df.columns if x not in y_cols_name]
    train_x = dcopy(df[x_cols_name])
    train_y = dcopy(df[y_cols_name])
    pos_cnt, neg_cnt = int(train_y.sum()), int(len(train_y) - train_y.sum())
    scipy_adj_matrix = nx.convert_matrix.to_scipy_sparse_matrix(sp_mul_G, format='coo')
    print('pos node cnts:', pos_cnt)
    print('neg node cnts:', neg_cnt, 'pos/all ratio:', pos_cnt / (pos_cnt + neg_cnt))

    adj = scipy_adj_matrix.tocsr()
    features = train_x.values
    labels = train_y.values.ravel()
    filename = PATH + 'bc' + str(int(SAMPLE_GSIZE / 10000)) + '.npz'
    np.savez(filename, adj_matrix=adj, node_attr=features, node_label=labels)


