import random
import pandas as pd
import torch
import numpy as np
import pickle
import os
import scipy.sparse as sp
from bisect import bisect_left
from pd import get_lgb_model
from sklearn import preprocessing

from graphgallery import functional as gf
from numba import jit, int32, int64

DP_MODELS = ['RobustGCN', 'SimPGCN']

DATASET_BLOCKCHAIN = ["bc" + str(i+1) for i in range(11)] + ['tedge', 'trans2vec']

# RES_COLUMNS = ['eva_asr', 'eva_asr_wl', 'poi_asr', 'poi_asr_wl', 'cost', 'embed_acc', 'average_atk_time', 'cluster_cost_time']

RES_COLUMNS = ['atked_model', 'eva_asr', 'poi_asr', 'cost', 'embed_acc', 'clean_acc', 'average_atk_time', 'cluster_cost_time']

RES_ERRORS = [-1 for _ in RES_COLUMNS]

# DATASETS = ['cora', 'citeseer', 'cora_full', 'citeseer_full', 'pubmed',
#             'flickr', 'coauthor_cs', 'coauthor_phy']

# EMBED_TYPE = ['MLP', 'GCN', 'SGC', 'PPNP', 'APPNP', 'SimPGCN',
#               'FastGCN', 'GraphMLP', 'GAT', 'ClusterGCN',
#               'GCN_E',
#               'DW', 'N2V', 'BANE']

DATASETS = ['cora', 'citeseer', 'chameleon', 'squirrel', 'cora_full', 'coauthor_phy', 'ogbn-arxiv']

EMBED_TYPE = ['MLP', 'SGC2', 'GCN2', 'FastGCN', 'SimPGCN']

ATTACKED_TYPE = ['GCN', 'SimPGCN', 'RobustGCN', 'GCN_Jaccard']

MIX_TYPE = [['MLP', 'SGC2'], ['MLP', 'GCN2']]

n_classes_dict = {
        'cora': 7,
        'cora_full': 70,
        'citeseer': 6,
        'ogbn-arxiv': 40,
        'reddit': 41,

        'coauthor_phy': 5,

        'chameleon': 5,
        'squirrel': 5,
        'blockchain30000': 2,
        'blockchain40000': 2,
        'blockchain50000': 2,
    }


def get_datasets(cmd_d):
    if len(cmd_d) <= 0:
        return DATASETS
    return cmd_d.split(',')


def get_embed_types(cmd_e):
    if len(cmd_e) <= 0:
        return EMBED_TYPE
    return cmd_e.split(',')


def get_mix_types(cmd_m):
    if len(cmd_m) <= 0:
        return MIX_TYPE
    return [mix.split(',') for mix in cmd_m.split('.')]


def get_attacked_types(cmd_a):
    if len(cmd_a) <= 0:
        return ATTACKED_TYPE
    return cmd_a.split(',')


def normalize_GCN(indices, weights, degree):
    row, col = indices
    inv_degree = torch.pow(degree, -0.5)
    normed_weights = weights * inv_degree[row] * inv_degree[col]

    return normed_weights


def to_list(targets):
    if np.ndim(targets) == 0:
        targets = [targets]
    elif isinstance(targets, np.ndarray):
        targets = targets.tolist()
    else:
        targets = list(targets)
    return targets


def to_array(targets):
    if np.ndim(targets) == 0:
        targets = np.array([targets])
    elif isinstance(targets, list):
        targets = np.array(targets)
    return targets


@jit(cache=True, nopython=True)
def get_purity(indices, indptr, labels):
    N = len(labels)
    purity = []
    for node_i in range(N):
        node_i_label = labels[node_i]
        nbrs = indices[indptr[node_i]:indptr[node_i + 1]]
        nbrs_label = labels[nbrs]
        cnt = (nbrs_label == node_i_label).mean()
        purity.append(cnt)
    return np.array(purity)


@jit(cache=True, nopython=True)
def get_purity_martix(purity, purity_r, labels):
    N = len(purity)
    purity_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                purity_matrix[i][j] = purity[i] * purity[j]
            else:
                purity_matrix[i][j] = purity_r[i] * purity_r[j]
    return purity_matrix


@jit(cache=True, nopython=True)
def get_purity_target_nbrs(target, nbrs, purity):
    return [purity[target] * purity[nbr] for nbr in nbrs]


@jit(cache=True, nopython=True)
def get_purity_list(indices, indptr, purity):
    N = len(purity)
    purity_list = []
    for node_i in range(N):
        nbrs = indices[indptr[node_i]:indptr[node_i + 1]]
        tmp_purity_list = [purity[node_i] * purity[nbr] for nbr in nbrs]
        purity_list.append(tmp_purity_list)
    return purity_list


@jit(cache=True, nopython=True)
def get_wl(indices, indptr, labels, wrong_label, eps):
    N = len(labels)
    wl = []
    wl_cnt = []
    for node_i in range(N):
        nbrs = indices[indptr[node_i]:indptr[node_i + 1]]
        nbrs_label = labels[nbrs]
        idx = nbrs_label == wrong_label
        wl.append(idx.mean())
        wl_cnt.append(idx.sum())
    return np.array(wl) + eps, np.log10((np.array(wl_cnt) + 10))


@jit(cache=True, nopython=True)
def get_wl_matrix(wl):
    N = len(wl)
    return np.array([wl[i] * wl[j] for i in range(N) for j in range(N)]).reshape((N, N))


@jit(cache=True, nopython=True)
def get_wl_target_nbrs(target, nbrs, wl):
    return [wl[target] * wl[nbr] for nbr in nbrs]


@jit(cache=True, nopython=True)
def get_wl_list(indices, indptr, wl):
    N = len(wl)
    wl_list = []
    for node_i in range(N):
        nbrs = indices[indptr[node_i]:indptr[node_i + 1]]
        tmp_wl_list = [wl[node_i] * wl[nbr] for nbr in nbrs]
        wl_list.append(tmp_wl_list)
    return wl_list


def roulette_wheel_selection(purity):
    '''
        Input: a list of N fitness values (list or tuple)
        Output: selected index
        '''
    purity = np.sort(purity)
    sumFits = sum(purity)
    # generate a random number
    rndPoint = random.uniform(0, sumFits)
    # calculate the accumulator: O(N)
    accumulator = []
    accsum = 0.0
    for pur in purity:
        accsum += pur
        accumulator.append(accsum)
    return bisect_left(accumulator, rndPoint)  # O(logN)


@jit(cache=True, nopython=True)
def stochastic_accept(purity):
    '''
    roulette wheel selections
    O(1) complexity
    paper: https://arxiv.org/pdf/1109.3627.pdf
    Input: a list of N purity values (list)
    Output: selected index
    '''
    N = len(purity)
    max_purity = max(purity)

    # select: O(1)
    while True:
        # randomly select an individual with uniform probability
        ind = int(N * np.random.random())
        # with probability w_i / w_max to accept the selection
        if np.random.random() <= purity[ind] / max_purity:
            return ind


def get_hop_neighbors(indices, indptr, target, hops=2):
    nodes, edges = get_hop_neighbors_njit(indices, indptr, target, hops)
    return np.unique(nodes), edges


@jit(cache=True, nopython=True)
def get_hop_neighbors_njit(indices, indptr, target, hops):
    edges = {}
    nodes = [np.int64(target)]
    start = 0
    for level in range(hops):
        length = len(nodes)
        for i in range(start, length):
            cur_node = nodes[i]
            nbrs = indices[indptr[cur_node]:indptr[cur_node + 1]].astype(np.int64)
            nodes.extend(nbrs)
            for nbr in nbrs:
                if (nbr, cur_node) not in edges:
                    edges[(cur_node, nbr)] = level
        start += length
    # nodes = np.unique(nodes)
    return nodes, edges


@jit(cache=True, nopython=True)
def get_hop_rate(walk_nodes, hop_nodes):
    intersection = np.intersect1d(hop_nodes, walk_nodes)
    # print(intersection.shape, intersection)
    return len(intersection) / len(hop_nodes), len(hop_nodes), len(walk_nodes)


@jit(cache=True, nopython=True)
def get_wrong_rate(nodes, wrong_label_nodes):
    intersection = np.intersect1d(nodes, wrong_label_nodes)
    return len(intersection) / len(wrong_label_nodes), len(intersection)


@jit(cache=True, nopython=True)
def cross_entropy(hi, hj):
    return (- np.sum(hi * np.log(hj)) - np.sum(hj * np.log(hi))) / 2


@jit(cache=True, nopython=True)
def get_cross_entropy_matrix(logits):
    N = len(logits)
    return np.array([cross_entropy(logits[i], logits[j]) for i in range(N) for j in range(N)]).reshape((N, N))


@jit(cache=True, nopython=True)
def get_cross_entropy_list(indices, indptr, logits):
    N = len(logits)
    cross_entropy_list = []
    for node_i in range(N):
        nbrs = indices[indptr[node_i]:indptr[node_i + 1]]
        tmp_cross_entropy_list = [cross_entropy(logits[node_i], logits[nbr]) for nbr in nbrs]
        cross_entropy_list.append(tmp_cross_entropy_list)
    return cross_entropy_list


@jit(cache=True, nopython=True)
def get_cross_entropy_target_nbrs(target, nbrs, logits):
    return np.array([cross_entropy(logits[target], logits[nbr]) for nbr in nbrs])


@jit(cache=True, nopython=True)
def get_target_subgraph_level(target, indices, indptr, N):
    start = 0
    seen = np.zeros(N) - 1
    seen[target] = 0
    targets = [target]
    level = 0
    while True:
        end = len(targets)
        while start < end:
            head = targets[start]
            nbrs = indices[indptr[head]:indptr[head + 1]]
            for u in nbrs:
                if seen[u] < 0:
                    targets.append(u)
                    seen[u] = level + 1
            start += 1
        level += 1
        if end == len(targets):
            break

    return seen


def save_test(_prefix, times, seeds):
    print(_prefix)
    tdf = None
    for i in range(times):
        filename = "_".join([_prefix, str(i)]) + '.csv'
        df = pd.read_csv(filename, index_col=0)
        if i == 0:
            tdf = df
        else:
            tdf += df
    tdf /= times
    tdf.index.name = ','.join(np.array(seeds).astype('str'))
    tdf.to_csv(_prefix + '_total.csv')


@jit(cache=True, nopython=True)
def get_purity_gains(indices, indptr, labels, purity, target, nbrs):
    first_nbrs = indices[indptr[target]:indptr[target + 1]]
    target_label = labels[target]
    purity_gains = []
    for nbr in nbrs:
        nbr_nbrs = indices[indptr[nbr]:indptr[nbr + 1]]  # 邻居的邻居
        nbr_nbrs_labels = labels[nbr_nbrs]  # 邻居的邻居的标签
        purity_cnt = (nbr_nbrs_labels == labels[nbr]).sum()
        nbr_cnt = len(nbr_nbrs)
        if nbr in first_nbrs:  # 一阶邻居，删边
            if target_label == labels[nbr]:
                purity_cnt -= 1.0
            nbr_cnt -= 1
        else:  # k阶邻居，加边
            if target_label == labels[nbr]:
                purity_cnt += 1.0
            nbr_cnt += 1
        # 防止0/0情况（nbr只有target一个邻居）
        if purity_cnt > 0:
            cur_purity = purity_cnt / nbr_cnt
            purity_gains.append([(cur_purity - purity[nbr]) / purity[nbr]])
        else:
            purity_gains.append([-1.0])

    return np.asarray(purity_gains)


@jit(cache=True, nopython=True)
def get_wl_gains(indices, indptr, labels, wl, target, nbrs, wrong_label):
    first_nbrs = indices[indptr[target]:indptr[target + 1]]
    wl_gains = []
    for nbr in nbrs:
        nbr_nbrs = indices[indptr[nbr]:indptr[nbr + 1]]  # 邻居的邻居
        nbr_nbrs_labels = labels[nbr_nbrs]  # 邻居的邻居的标签
        wl_cnt = (nbr_nbrs_labels == wrong_label).sum()
        nbr_cnt = len(nbr_nbrs)
        if nbr in first_nbrs:  # 一阶邻居，删边
            nbr_cnt -= 1
        else:  # k阶邻居，加边
            nbr_cnt += 1
        # 防止0/0情况（nbr只有target一个邻居）
        if nbr_cnt > 0:
            cur_wl = wl_cnt / nbr_cnt
            wl_gains.append([(cur_wl - wl[nbr]) / wl[nbr]])
        else:
            wl_gains.append([0.0])

    return np.asarray(wl_gains)


@jit(cache=True, nopython=True)
def random_choice(arr, p):
    """Similar to `numpy.random.choice` and it suppors p=option in numba.
    refer to <https://github.com/numba/numba/issues/2539#issuecomment-507306369>

    Parameters
    ----------
    arr : 1-D array-like
    p : 1-D array-like
        The probabilities associated with each entry in arr

    Returns
    -------
    sample : ndarray with 1 element
        The generated random sample
    """
    return arr[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]


def load_pickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def get_pd(attacked_model, args):
    embedded_features = attacked_model.predict(args.train_nodes)
    original_features = args.node_attr
    true_labels = args.node_label
    combined_features_labels = np.hstack((original_features, embedded_features, true_labels.reshape(-1,1)))

    columns_name = ['f%02d' % i for i in range(combined_features_labels.shape[1] - 1)] + ['label']
    df_combined = pd.DataFrame(data=combined_features_labels, columns=columns_name, dtype=float)
    df_combined['label'] = df_combined['label'].astype(int)

    y_cols_name = ['label']
    x_cols_name = [x for x in df_combined.columns if x not in y_cols_name]

    x = df_combined[x_cols_name]
    y = df_combined[y_cols_name]
    test_res, all_predict, lgb_model = get_lgb_model(x, y, args.seed)
    # print(test_res)
    return all_predict, lgb_model


def get_train_x(attacked_model, args, target):
    embedded_features = attacked_model.predict(args.train_nodes)
    original_features = args.node_attr
    combined_features_labels = np.hstack((original_features, embedded_features))

    columns_name = ['f%02d' % i for i in range(combined_features_labels.shape[1])]
    df_combined = pd.DataFrame(data=combined_features_labels, columns=columns_name, dtype=float)

    x_cols_name = [x for x in df_combined.columns if x != 'label']
    train_x = df_combined[x_cols_name].iloc[[target]]
    return train_x


def get_wrong_labels(logits, targets, labels):
    wrong_labels = []
    for target in targets:
        logit = logits[target]
        idx = list(set(range(logit.size)) - set([labels[target]]))
        wrong_label = idx[logit[idx].argmax()]
        wrong_labels.append(wrong_label)
    wrong_labels = np.array(wrong_labels)
    return wrong_labels


# compute the max pro class of cluster nodes
def mapCluster2GCN(targets, labels, perturbed_nodes):
    gcn_labels = []
    for i, target in enumerate(targets):
        labels_i = labels[perturbed_nodes[i][1]]
        distribution = {}
        ui = list(set(labels_i))
        for li in ui:
            distribution[li] = (labels_i == li).sum()
        distribution = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        best = distribution[0]
        gcn_labels.append((best[0], best[1] / len(labels_i)))
    gcn_labels = np.array(gcn_labels)
    return gcn_labels[:, 0], gcn_labels[:, 1]


HIDS = [[512, 256, 128, 64],
        [512, 256, 128, 64, 32],
        [512, 256, 128, 64, 32, 16]]
HIDS2 = [[128, 64],
         [512, 256, 128, 64, 32, 16]]
HIDS3 = [
        [64],
        [128, 64],
        [256, 128, 64],
    ]
CORA = 'cora'
CITESEER = 'citeseer'
CHAMELEON = 'chameleon'
SQUIRREL = 'squirrel'
CORA_FULL = 'cora_full'
COAUTHOR_PHY = 'coauthor_phy'
OGBN_ARXIV = 'ogbn-arxiv'
REDDIT = 'reddit'

EMBED_SGC = 'sgc2'
EMBED_GCN = 'gcn2'
EMBED_FGCN = 'fastgcn'
EMBED_MLP = 'mlp'

ATKED_GCN = 'gcn'
ATKED_RGCN = 'robustgcn'
ATKED_JGCN = 'gcn_jaccard'
ATKED_SIMPGCN = 'simpgcn'

MODEL_PARAMS = {
    # multi layers
    CORA: {
        EMBED_FGCN: {
            ATKED_GCN: [HIDS3[2], 5e-5, 5e-2],
            ATKED_JGCN: [HIDS3[1], 5e-5, 1e-2],
            ATKED_RGCN: [HIDS3[1], 5e-5, 1e-2],
            ATKED_SIMPGCN: [HIDS3[1], 5e-3, 1e-2],
        },
        EMBED_GCN: {
            ATKED_GCN: [HIDS3[1], 5e-5, 5e-2],
            ATKED_JGCN: [HIDS3[2], 5e-4, 1e-2],
            ATKED_RGCN: [HIDS3[1], 5e-4, 1e-2],
            ATKED_SIMPGCN: [HIDS3[1], 5e-3, 5e-2],
        },
        EMBED_MLP: {
            ATKED_GCN: [[1024, 1024, 512, 512, 256, 256, 128, 64], 5e-5, 1e-3],
            ATKED_JGCN: [[1024, 1024, 512, 512, 256, 256, 128, 64], 5e-5, 1e-3],
            ATKED_RGCN: [[1024, 1024, 512, 512, 256, 256, 128, 64], 5e-5, 1e-2],
            ATKED_SIMPGCN: [[1024, 1024, 512, 512, 256, 256, 128, 64], 5e-4, 1e-2],
        },
        EMBED_SGC: {
            ATKED_GCN: [HIDS3[1], 5e-5, 5e-2],
            ATKED_JGCN: [HIDS3[0], 5e-5, 5e-3],
            ATKED_RGCN: [HIDS3[2], 5e-4, 1e-3],
            ATKED_SIMPGCN: [HIDS3[1], 5e-3, 5e-2],
        },
    },
    CITESEER: {
        EMBED_FGCN: {
            ATKED_GCN: [HIDS2[0], 5e-3, 5e-3],
            ATKED_JGCN: [HIDS2[0], 5e-3, 5e-3],
            ATKED_RGCN: [HIDS2[0], 5e-4, 1e-3],
            ATKED_SIMPGCN: [HIDS2[0], 5e-5, 1e-2],
        },
        EMBED_GCN: {
            ATKED_GCN: [HIDS2[0], 5e-4, 1e-2],
            ATKED_JGCN: [HIDS2[0], 5e-2, 1e-2],
            ATKED_RGCN: [HIDS2[0], 5e-4, 1e-3],
            ATKED_SIMPGCN: [HIDS2[0], 5e-4, 1e-2],
        },
        EMBED_MLP: {
            ATKED_GCN: [HIDS2[0], 5e-5, 5e-2],
            ATKED_JGCN: [HIDS2[0], 5e-4, 5e-3],
            ATKED_RGCN: [HIDS2[0], 5e-5, 5e-2],
            ATKED_SIMPGCN: [HIDS2[0], 5e-2, 5e-2],
        },
        EMBED_SGC: {
            ATKED_GCN: [HIDS2[0], 5e-2, 1e-2],
            ATKED_JGCN: [HIDS2[0], 5e-2, 1e-3],
            ATKED_RGCN: [HIDS2[0], 5e-2, 1e-2],
            ATKED_SIMPGCN: [HIDS2[0], 5e-4, 5e-2],
        },
    },
    # 2 layers
    CHAMELEON: {
        EMBED_FGCN: {
            ATKED_GCN: [HIDS2[0], 5e-5, 5e-3],
            ATKED_JGCN: [HIDS2[0], 5e-5, 5e-3],
            ATKED_RGCN: [HIDS2[0], 5e-5, 5e-3],
            ATKED_SIMPGCN: [HIDS2[0], 5e-5, 5e-3],
        },
        EMBED_GCN: {
            ATKED_GCN: [HIDS2[0], 5e-3, 1e-2],
            ATKED_JGCN: [HIDS2[0], 5e-3, 1e-3],
            ATKED_RGCN: [HIDS2[0], 5e-3, 1e-2],
            ATKED_SIMPGCN: [HIDS2[0], 5e-5, 1e-2],
        },
        EMBED_MLP: {
            ATKED_GCN: [HIDS2[0], 5e-3, 5e-3],
            ATKED_JGCN: [HIDS2[0], 5e-2, 1e-2],
            ATKED_RGCN: [HIDS2[0], 5e-3, 5e-3],
            ATKED_SIMPGCN: [HIDS2[0], 5e-3, 1e-2],
        },
        EMBED_SGC: {
            ATKED_GCN: [HIDS2[0], 5e-5, 5e-2],
            ATKED_JGCN: [HIDS2[0], 5e-3, 5e-3],
            ATKED_RGCN: [HIDS2[0], 5e-2, 5e-3],
            ATKED_SIMPGCN: [HIDS2[0], 5e-3, 5e-3],
        },
    },
    SQUIRREL: {
        EMBED_FGCN: {
            ATKED_GCN: [HIDS2[0], 5e-2, 5e-3],
            ATKED_JGCN: [HIDS2[0], 5e-4, 5e-3],
            ATKED_RGCN: [HIDS2[0], 5e-5, 1e-2],
            ATKED_SIMPGCN: [HIDS2[0], 5e-2, 1e-3],
        },
        EMBED_GCN: {
            ATKED_GCN: [HIDS2[0], 5e-2, 1e-3],
            ATKED_JGCN: [HIDS2[0], 5e-2, 5e-3],
            ATKED_RGCN: [HIDS2[0], 5e-2, 5e-3],
            ATKED_SIMPGCN: [HIDS2[0], 5e-5, 1e-2],
        },
        EMBED_MLP: {
            ATKED_GCN: [HIDS2[0], 5e-4, 5e-2],
            ATKED_JGCN: [HIDS2[0], 5e-4, 1e-2],
            ATKED_RGCN: [HIDS2[0], 5e-2, 1e-3],
            ATKED_SIMPGCN: [HIDS2[0], 5e-2, 5e-2],
        },
        EMBED_SGC: {
            ATKED_GCN: [HIDS2[0], 5e-3, 5e-3],
            ATKED_JGCN: [HIDS2[0], 5e-3, 1e-3],
            ATKED_RGCN: [HIDS2[0], 5e-3, 5e-3],
            ATKED_SIMPGCN: [HIDS2[0], 5e-3, 5e-3],
        },
    },
    # CORA_FULL: {
    #     EMBED_FGCN: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_GCN: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_MLP: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_SGC: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    # },
    # COAUTHOR_PHY: {
    #     EMBED_FGCN: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_GCN: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_MLP: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_SGC: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    # },
    # OGBN_ARXIV: {
    #     EMBED_FGCN: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_GCN: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_MLP: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_SGC: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    # },
    # REDDIT: {
    #     EMBED_FGCN: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_GCN: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_MLP: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    #     EMBED_SGC: {
    #         ATKED_GCN: [],
    #         ATKED_JGCN: [],
    #         ATKED_RGCN: [],
    #         ATKED_SIMPGCN: [],
    #     },
    # },
}


def get_model_parms(dataset, embed_model, atked_model):
    dataset = dataset.lower()
    embed_model = embed_model.lower()
    atked_model = atked_model.lower()
    parms = MODEL_PARAMS[dataset][embed_model][atked_model]
    assert len(parms) < 4, 'get_model_parms error'
    return parms[0], ['relu' for _ in parms[0]] if embed_model != "sgc2" else [None for _ in parms[0]], parms[1], parms[2]


def accuracy(output, labels):
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))


def _normalize_adj(adj, power=-1/2, device="cpu"):
    """Row-normalize sparse matrix"""
    adj = sp.csr_matrix(adj).todense()
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    A = adj.to(device) + torch.eye(len(adj)).to(device)
    D_power = (A.sum(1)).pow(power)
    D_power[torch.isinf(D_power)] = 0.
    D_power = torch.diag(D_power)
    return D_power @ A @ D_power


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    # num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def normalize_sparse_tensor(adj, fill_value=1):
    """Normalize sparse tensor. Need to import torch_scatter
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes= adj.size(0)
    edge_index, edge_weight = add_self_loops(
	edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)


def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
    """
    device = adj.device
    if sparse:
        # warnings.warn('If you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')
        # TODO if this is too slow, uncomment the following code,
        # but you need to install torch_scatter
        return normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx


def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def _normalize_adj_simpgcn(adj, features, device, normalize=True):
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))
    if normalize:
        if is_sparse_tensor(adj):
            adj_norm = normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = normalize_adj_tensor(adj)
    else:
        adj_norm = adj
    return adj_norm.to(device), features.to(device)

we_lr_dic = {
    '5e-5':'5e-5',
    '5e-05':'5e-5',
    '0.05':'5e-2',
    '0.005':'5e-3',
    '0.0005':'5e-4',
    '0.1':'1e-1',
    '0.01':'1e-2',
    '0.001':'1e-3',
    '0.0001':'1e-4'
}


def mhash(l):
    s = ""
    for ti in l:
        if isinstance(ti, list):
            for tii in ti:
                s += str(tii)
        else:
            s += we_lr_dic[str(ti)]
    return s


def get_split_atked_types(dataset, embed_type, atked_types, not_split=False):
    if not_split:
        return [atked_types]
    split_atked_types = {}
    dataset = dataset.lower()
    embed_type = embed_type.lower()
    for i, atked_type in enumerate(atked_types):
        cur_hash = mhash(MODEL_PARAMS[dataset][embed_type][atked_type.lower()])
        if cur_hash not in split_atked_types:
            split_atked_types[cur_hash] = [atked_type]
        else:
            split_atked_types[cur_hash].append(atked_type)

    return list(split_atked_types.values())


def to_tensor(adj=None, features=None, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """
    is_adj = adj is not None
    is_features = features is not None
    is_labels = labels is not None
    if is_adj:
        if sp.issparse(adj):
            adj = sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = torch.FloatTensor(adj)
        adj = adj.to(device)
    if is_features:
        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features)
        elif type(features) is not torch.Tensor:
            features = torch.FloatTensor(np.array(features))
        features = features.to(device)
    if is_labels:
        labels = torch.LongTensor(labels).to(device)

    if is_adj and is_features and is_labels:
        return adj, features, labels

    if is_adj and is_features and not is_labels:
        return adj, features

    if is_adj and not is_features and is_labels:
        return adj, labels

    if not is_adj and is_features and is_labels:
        return features, labels

    if is_adj and not is_features and not is_labels:
        return adj

    if not is_adj and is_features and not is_labels:
        return features

    if not is_adj and not is_features and is_labels:
        return labels

    assert False, "adj and features and labels all None"


def get_remain_ettt(filename, cur_edges, cur_result, del_key_split_idx):
    # tedge, trans2vec
    # del_key = methodType_atkedModel_embedType, such as MLP_SVM_TBS+WBS
    # bmgcn, normal, jiaying
    # del_key = methodType_atkedModel, such as MLP_GCN
    if '.csv' not in filename:
        filename += '.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0)
        idx = list(df.index)
        for i in idx:
            cur_result.loc[i] = df.loc[i]
            split = np.array(i.split('_'))
            split = split[del_key_split_idx]
            del_key = "_".join(split)
            cur_edges.pop(del_key, None)
    return list(cur_edges.keys())