import random

import torch
import numpy as np
from bisect import bisect_left
from graphgallery import functional as gf
from numba import njit


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


def get_purity(adj, labels):
    indices = adj.indices
    indptr = adj.indptr
    N = adj.shape[0]
    purity = []
    for node_i in range(N):
        node_i_label = labels[node_i]
        nbrs = indices[indptr[node_i]:indptr[node_i+1]]
        nbrs_label = labels[nbrs]
        cnt = (nbrs_label == node_i_label).mean()
        purity.append(cnt)
    return np.array(purity)


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


def get_wl(adj, labels, wrong_label, eps):
    indices = adj.indices
    indptr = adj.indptr
    N = adj.shape[0]
    wl = []
    wl_cnt = []
    for node_i in range(N):
        nbrs = indices[indptr[node_i]:indptr[node_i + 1]]
        nbrs_label = labels[nbrs]
        idx = nbrs_label == wrong_label
        wl.append(idx.mean())
        wl_cnt.append(idx.sum())
    return np.array(wl) + eps, np.log10((np.array(wl_cnt) + 10))


def get_wl_matrix(wl):
    N = len(wl)
    return np.array([wl[i] * wl[j] for i in range(N) for j in range(N)]).reshape((N, N))

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


def get_hop_neighbors(adj_matrix, target, hops=2):
    indices = adj_matrix.indices
    indptr = adj_matrix.indptr

    edges = {}
    nodes = [target]
    start = 0
    for level in range(hops):
        length = len(nodes)
        for i in range(start, length):
            cur_node = nodes[i]
            # print(cur_node)
            nbrs = indices[indptr[cur_node]:indptr[cur_node + 1]]
            nodes.extend(nbrs)
            for nbr in nbrs:
                if (nbr, cur_node) not in edges:
                    edges[(cur_node, nbr)] = level
        start += length

    return np.unique(nodes), edges


def get_hop_rate(walk_nodes, hop_nodes):
    intersection = np.intersect1d(hop_nodes, walk_nodes)
    # print(intersection.shape, intersection)
    return len(intersection) / len(hop_nodes), len(hop_nodes), len(walk_nodes)


def get_wrong_rate(nodes, wrong_label_nodes):
    intersection = np.intersect1d(nodes, wrong_label_nodes)
    return len(intersection) / len(wrong_label_nodes), len(intersection)


@njit
def cross_entropy(hi, hj):
    return (- np.sum(hi * np.log(hj)) - np.sum(hj * np.log(hi))) / 2


@njit
def get_cross_entropy_matrix(logits):
    N = len(logits)
    cem = []
    for i in range(N):
        ces = []
        for j in range(N):
            ce = cross_entropy(logits[i], logits[j])
            ces.append(ce)
        cem.append(ces)
    return np.array(cem)