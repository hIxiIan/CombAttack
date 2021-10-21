import random
import pandas as pd
import torch
import numpy as np
from bisect import bisect_left
from graphgallery import functional as gf
from numba import njit, int32, int64


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


@njit
def get_purity(indices, indptr, labels):
    N = len(labels)
    purity = []
    for node_i in range(N):
        node_i_label = labels[node_i]
        nbrs = indices[indptr[node_i]:indptr[node_i+1]]
        nbrs_label = labels[nbrs]
        cnt = (nbrs_label == node_i_label).mean()
        purity.append(cnt)
    return np.array(purity)


@njit
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


@njit
def get_purity_target_nbrs(target, nbrs, purity):
    return [purity[target] * purity[nbr] for nbr in nbrs]


@njit
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


@njit
def get_wl_matrix(wl):
    N = len(wl)
    return np.array([wl[i] * wl[j] for i in range(N) for j in range(N)]).reshape((N, N))


@njit
def get_wl_target_nbrs(target, nbrs, wl):
    return [wl[target] * wl[nbr] for nbr in nbrs]


@njit
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
        ind = int(N * random.random())
        # with probability w_i / w_max to accept the selection
        if random.random() <= purity[ind] / max_purity:
            return ind


def get_hop_neighbors(indices, indptr, target, hops=2):
    nodes, edges = get_hop_neighbors_njit(indices, indptr, target, hops)
    return np.unique(nodes), edges


@njit
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


@njit
def get_hop_rate(walk_nodes, hop_nodes):
    intersection = np.intersect1d(hop_nodes, walk_nodes)
    # print(intersection.shape, intersection)
    return len(intersection) / len(hop_nodes), len(hop_nodes), len(walk_nodes)


@njit
def get_wrong_rate(nodes, wrong_label_nodes):
    intersection = np.intersect1d(nodes, wrong_label_nodes)
    return len(intersection) / len(wrong_label_nodes), len(intersection)


@njit
def cross_entropy(hi, hj):
    return (- np.sum(hi * np.log(hj)) - np.sum(hj * np.log(hi))) / 2


@njit
def get_cross_entropy_matrix(logits):
    N = len(logits)
    return np.array([cross_entropy(logits[i], logits[j]) for i in range(N) for j in range(N)]).reshape((N, N))


@njit
def get_cross_entropy_target_nbrs(target, nbrs, logits):
    return [cross_entropy(logits[target], logits[nbr]) for nbr in nbrs]


@njit
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


def save_test(_prefix, times):
    tdf = None
    for i in range(times):
        filename = "_".join([_prefix, str(i)]) + '.csv'
        df = pd.read_csv(filename, index_col=0)
        if i == 0:
            tdf = df
        else:
            tdf += df
    tdf /= times
    tdf.to_csv(_prefix + '_total.csv')