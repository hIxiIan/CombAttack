import sys
import networkx as nx
import pickle
import numpy as np
import time
import random
import argparse
import torch
import os
import pandas as pd
from gensim.models import Word2Vec
from time import strftime, localtime
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import train_test_split
from numba import jit, njit
from utils import load_pickle


METHOD_MAP = {
    'TEDGE': ['time_uniform', 'time_uniform', 'amount_uniform', 1.0],
    'TBS': ['time_close_linear', 'time_close_linear', 'amount_uniform', 1.0],
    'WBS': ['time_uniform', 'time_uniform', 'amount_linear', 0.0],
    'TBS+WBS': ['time_close_linear', 'time_close_linear', 'amount_linear', 0.5],
}


@njit
def numba_seed(sd):
    np.random.seed(sd)


def random_seed(seed=None):
    np.random.seed(seed)
    numba_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(original_array):
    x = np.array(original_array)
    max = np.max(x)
    return np.exp(x-max) / np.sum(np.exp(x-max))


def tanh(original_array):
    x = np.array(original_array)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


@jit(cache=True, nopython=True)
def weight_choice(unnormalized_probs):
    norm_const = np.sum(unnormalized_probs)
    normalized_probs = np.array([float(u_prob / norm_const) for u_prob in unnormalized_probs])  # 归一化
    J, q = alias_setup(normalized_probs)
    idx = alias_draw(J, q)
    return idx


@jit(cache=True, nopython=True)
def random_weight_choice(p):
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
    return np.searchsorted(np.cumsum(p), np.random.random(), side="right")


def make_redundancy(arr):
    new_arr = []
    col = 0
    for a in arr:
        col = max(col, len(a))
    for a in arr:
        new_arr.append(list(a) + [-1] * (col - len(a)))
    return new_arr


@jit(cache=True, nopython=True)
def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


@jit(cache=True, nopython=True)
def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)
    kk = np.random.randint(K)
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def linear_rank_mapping(original_array, order='ascending'):
    x = np.array(original_array)
    if order == 'ascending':
        return (np.argsort(x) + 1)
    elif order == 'descending':
        return (np.argsort(-x) + 1)
        # return (x.argsort() + 1)


@jit(cache=True, nopython=True)
def normalized_probs(unnormalized_probs):
    if len(unnormalized_probs) > 0:  # 有符合条件的下一个点
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()

    return normalized_probs


@jit(cache=True, nopython=True)
def combine_probs(p1, p2, alpha):
    probs1 = normalized_probs(p1)
    probs2 = normalized_probs(p2)

    if len(probs1) != len(probs2):
        print("ERROR", "len(probs1) != len(probs2)")

    combine_probs = np.multiply(np.power(probs1, alpha), np.power(probs2, 1 - alpha))

    return combine_probs


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size + 1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num, "load_embeddings error"
    return vectors


def load_labels(filename):
    fin = open(filename, 'r')
    labels = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        node = str(int(vec[0]) - 1)
        labels[node] = int(vec[1])
    fin.close()
    return labels


class tGraph(object):
    def __init__(self, file_, DiG=None, filetype='txt_f', output_Gpkl=False, verbose=0):
        self.G = nx.MultiDiGraph()
        self.min_time = sys.maxsize
        self.max_time = 0
        self.min_amount = sys.maxsize
        self.max_amount = 0

        if verbose > 0:
            print("Loading file", file_, "...")
        edge_key = 0

        if filetype == "txt_f":
            with open(file_) as f:
                for l in f:
                    x, y, a, t = l.strip().split(',')
                    a = float(a)
                    t = int(t)
                    x = str(int(x) - 1)
                    y = str(int(y) - 1)
                    if self.G.has_edge(x, y, t):
                        if self.G[x][y][t]['weight'] != a:
                            self.G[x][y][t]['weight'] += a
                    else:
                        self.G.add_edge(x, y, key=t, weight=a)
                    edge_key = edge_key + 1

                    if t < self.min_time:
                        self.min_time = t
                    elif t > self.max_time:
                        self.max_time = t
        elif filetype == "pd_f":
            for ind, edge in enumerate(list(set(nx.edges(DiG)))):
                (u, v) = edge
                egs = DiG[u][v]
                for egi in egs:
                    eg = egs[egi]
                    a, t = float(eg['amount']), int(eg['timestamp'])
                    x = str(u)
                    y = str(v)
                    if self.G.has_edge(x, y, t):
                        if self.G[x][y][t]['weight'] != a:
                            self.G[x][y][t]['weight'] += a
                    else:
                        self.G.add_edge(x, y, key=t, weight=a)
                    edge_key = edge_key + 1

                    if t < self.min_time:
                        self.min_time = t
                    elif t > self.max_time:
                        self.max_time = t

        self.number_of_nodes = self.G.number_of_nodes()
        self.number_of_edges = self.G.number_of_edges()
        if verbose > 0:
            print("Summary of graph:")
            print("Number of nodes: ", self.number_of_nodes)
            print("Number of edges: ", self.number_of_edges)
            print("Number of edge_key: ", edge_key)
            print("Min time: ", self.min_time)
            print("Max time: ", self.max_time)


class tGraphNE(object):
    def __init__(self, tG, time_biased_type, first_biased_type, amount_biased, alpha, output,
                 dimensions=128, num_walks=4, walk_length=10, output_pklG=False,
                 window_size=4, workers=1, hs=1, seed=2022, verbose=0, save_features=True, is_dan=True, rac=False):
        self.G = tG.G
        self.min_time = tG.min_time
        self.max_time = tG.max_time
        self.verbose = verbose
        self.weight_choice = weight_choice
        if rac:
            self.weight_choice = random_weight_choice

        self.time_biased_type = time_biased_type  # choice = "unbiased", "amount-weighted" "linear", "exp"
        self.first_biased_type = first_biased_type
        self.amount_biased = amount_biased
        self.alpha = alpha
        t1 = time.time()
        if is_dan:
            walks = self.dan_simulate_walks(num_walks, walk_length)  # 随机游走
        else:
            walks = self.simulate_walks(num_walks, walk_length)  # 随机游走
        t2 = time.time()
        word2vec_model = Word2Vec(sentences=walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, hs=1,
                                  workers=workers, seed=seed)
        t3 = time.time()
        features = word2vec_model.wv.vectors[np.fromiter(map(int, word2vec_model.wv.index_to_key), np.int32).argsort()]
        self.features = features
        self.walks = walks
        self.word2vec_model = word2vec_model

        if save_features:
            pd.DataFrame(features).to_csv(output, index=None)

        if verbose > 0:
            print('features.shape:{}'.format(features.shape))
            print("Walking time:", t2 - t1)
            print("Learn embeddings time:", t3 - t2)
            print("Embeddings are saved in ", output)

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        对每个结点，根据num_walks得出其多条随机游走路径

        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        if self.verbose > 0:
            print("Walk iteration:")
        for walk_iter in range(num_walks):
            if self.verbose > 0:
                print(str(walk_iter + 1), '/', str(num_walks))
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.temporal_walk(walk_length=walk_length, start_node=node))
        return walks

    def temporal_walk(self, walk_length, start_node):
        """
        功能： 从一个初始结点计算一个随机游走
        输入：
        walk_length: 随机游走序列长度
        start_node: 初始结点
        返回：
        列表，随机游走序列
        """
        G = self.G
        walk = [start_node]  # 类型：list
        walk_edge = []
        walk_time = []  ##类型：list, 大小比walk的小1
        # walk_key = []

        cur = start_node
        cur_nbrs = np.array(list(G.neighbors(cur)))
        nbr_keys = []
        for nbr in cur_nbrs:
            nbr_key = list(G.get_edge_data(cur, nbr))  # cur领边的key数组
            nbr_keys.append(nbr_key)
        nbr_keys = np.array(make_redundancy(nbr_keys), dtype=np.int32)
        if len(cur_nbrs) == 0:
            return walk

        unnormalized_probs_t, tmp_node, tmp_time, tmp_key = get_first_step(cur_nbrs, nbr_keys, self.first_biased_type, self.max_time, self.min_time)
        if unnormalized_probs_t is not None and len(unnormalized_probs_t) > 0:
            selected = weight_choice(unnormalized_probs_t)
            walk.append(tmp_node[selected])
            walk_time.append(tmp_time[selected])
            walk_edge.append(tmp_key[selected])
        else:
            return walk

        while len(walk) < walk_length:
            prevtime = walk_time[-1]
            cur = walk[-1]  # 名为walk的list的最后一个元素，当前游走到的结点
            cur_nbrs = np.array(list(G.neighbors(cur)))
            weight_keys = []
            nbr_keys = []
            for nbr in cur_nbrs:
                nbr_key = list(G.get_edge_data(cur, nbr))  # cur领边的key数组
                nbr_keys.append(nbr_key)
                weight_keys.append([G[cur][nbr][nk]['weight'] for nk in nbr_key])
            nbr_keys = np.array(make_redundancy(nbr_keys), dtype=np.int32)
            weight_keys = np.array(make_redundancy(weight_keys), dtype=np.float32)
            if len(cur_nbrs) == 0 or len(nbr_keys) == 0:
                break

            unnormalized_probs_t, tmp_node, tmp_time, tmp_key = get_next_step(cur_nbrs, nbr_keys, weight_keys, self.time_biased_type, self.amount_biased, self.max_time, self.max_time, prevtime)
            if unnormalized_probs_t is not None and len(unnormalized_probs_t) > 0:
                selected = weight_choice(unnormalized_probs_t)
                walk.append(tmp_node[selected])
                walk_time.append(tmp_time[selected])
                walk_edge.append(tmp_key[selected])
            else:
                break
        return walk

    def dan_simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        对每个结点，根据num_walks得出其多条随机游走路径

        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        if self.verbose > 0:
            print("Walk iteration:")
        for walk_iter in range(num_walks):
            if self.verbose > 0:
                print(str(walk_iter + 1), '/', str(num_walks))
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.dan_temporal_walk(walk_length=walk_length, start_node=node))
        return walks

    def dan_temporal_walk(self, walk_length, start_node):
        """
        功能： 从一个初始结点计算一个随机游走
        输入：
        walk_length: 随机游走序列长度
        start_node: 初始结点
        返回：
        列表，随机游走序列
        """

        walk = [start_node]  # 类型：list
        walk_edge = []
        walk_time = []  ##类型：list, 大小比walk的小1
        # walk_key = []

        cur = start_node
        next_node, next_time, next_key = self.get_first_step(cur)
        if next_node is not None:
            walk.append(next_node)
            walk_time.append(next_time)
            walk_edge.append(next_key)
        else:
            return walk

        while len(walk) < walk_length:
            prevtime = walk_time[-1]
            cur = walk[-1]  # 名为walk的list的最后一个元素，当前游走到的结点
            next_node, next_time, next_key = self.get_next_step(cur, prevtime)
            if next_node is not None:
                walk.append(next_node)
                walk_time.append(next_time)
                walk_edge.append(next_key)
            else:
                break
        return walk

    def get_first_step(self, cur):
        G = self.G
        tmp_key = []
        tmp_node = []
        tmp_time = []
        unnormalized_probs_t = []

        cur_nbrs = list(G.neighbors(cur))
        if self.time_biased_type == "simple_graph":  # DeepWalk
            for nbr in cur_nbrs:
                tmp_node.append(nbr)
                unnormalized_probs_t.append(1)

            if len(unnormalized_probs_t) > 0:
                idx = weight_choice(unnormalized_probs_t)
                next_node = tmp_node[idx]
                next_time = 0
                next_key = 0
                return next_node, next_time, next_key
            else:
                return None, None, None  # 没有符合条件的

        else:
            for nbr in cur_nbrs:
                nbr_key = list(G.get_edge_data(cur, nbr))  # cur领边的key数组
                for k in nbr_key:
                    t = k
                    if self.first_biased_type == "time_uniform":
                        unnormalized_probs_t.append(1)
                    elif self.first_biased_type == "time_freq":
                        unnormalized_probs_t.append(self.max_time - t + 1)
                    elif self.first_biased_type == "time_close_linear":
                        unnormalized_probs_t.append(self.max_time - t + 1)
                    elif self.first_biased_type == "time_far":
                        unnormalized_probs_t.append(t - self.min_time + 1)
                    elif self.first_biased_type == "time_far_linear":
                        unnormalized_probs_t.append(t)

                    tmp_node.append(nbr)
                    tmp_time.append(t)
                    tmp_key.append(k)

            if self.first_biased_type == "time_close_linear":  # TBS descending
                unnormalized_probs_t = linear_rank_mapping(unnormalized_probs_t, order='descending')
            elif self.first_biased_type == "time_far_linear":  # TBS ascending
                unnormalized_probs_t = linear_rank_mapping(unnormalized_probs_t)

            if len(unnormalized_probs_t) > 0:  # 有符合条件的下一个点
                unnormalized_probs_t = np.asarray(unnormalized_probs_t)
                selected = self.weight_choice(unnormalized_probs_t)
                next_node = tmp_node[selected]
                next_time = tmp_time[selected]
                next_key = tmp_key[selected]
                return next_node, next_time, next_key
            else:
                return None, None, None  # 没有符合条件的

    def get_next_step(self, cur, prevtime=0):
        """
        功能：给定一个当前随机游走到的结点cur，这个两个相连的结点（可能有多条边），得出
        输出：
        #return J, q
        直接输出下一个节点，以及时间戳
        """
        G = self.G

        tmp_key = []
        tmp_node = []
        tmp_time = []
        unnormalized_probs_t = []
        unnormalized_probs_a = []

        cur_nbrs = list(G.neighbors(cur))
        if self.time_biased_type == "simple_graph":  # DeepWalk
            for nbr in cur_nbrs:
                tmp_node.append(nbr)
                unnormalized_probs_t.append(1)

            if len(unnormalized_probs_t) > 0:
                idx = weight_choice(unnormalized_probs_t)
                next_node = tmp_node[idx]
                next_time = 0
                next_key = 0
                return next_node, next_time, next_key
            else:
                return None, None, None  # 没有符合条件的
        else:
            for nbr in cur_nbrs:
                nbr_key = list(G.get_edge_data(cur, nbr))  # cur领边的key数组
                for k in nbr_key:
                    t = k
                    a = G[cur][nbr][k]['weight']
                    if self.time_biased_type == "no_time_limit":
                        unnormalized_probs_t.append(1)

                    elif t >= prevtime:
                        unnormalized_probs_a.append(a)

                        if self.time_biased_type == "time_uniform":
                            unnormalized_probs_t.append(1)
                        elif self.time_biased_type == "time_close_raw":
                            unnormalized_probs_t.append(self.max_time - t + 1)
                        elif self.time_biased_type == "time_close_exp":
                            unnormalized_probs_t.append(t - prevtime)
                        else:
                            unnormalized_probs_t.append(t - prevtime + 1)
                        tmp_time.append(t)
                        tmp_node.append(nbr)
                        tmp_key.append(k)

            if self.time_biased_type == "time_close_linear":  # TBS descending
                unnormalized_probs_t = linear_rank_mapping(unnormalized_probs_t, order='descending')
            elif self.time_biased_type == "time_far_linear":  # TBS ascending
                unnormalized_probs_t = linear_rank_mapping(unnormalized_probs_t)
            elif self.time_biased_type == "time_freq_tanh":
                unnormalized_probs_t = tanh(unnormalized_probs_t)
            elif self.time_biased_type == "time_close_exp":
                unnormalized_probs_t = softmax(unnormalized_probs_t)

            # 金额偏好，映射函数缓解过小权重几乎没用
            if self.amount_biased == "amount_linear":  # WBS ascending
                unnormalized_probs_a = linear_rank_mapping(unnormalized_probs_a)
            elif self.amount_biased == "amount_tanh":
                unnormalized_probs_a = tanh(unnormalized_probs_a)
            elif self.amount_biased == "amount_exp":
                unnormalized_probs_a = softmax(unnormalized_probs_a)

            if len(unnormalized_probs_t) > 0:  # 有符合条件的下一个点
                unnormalized_probs_t = np.asarray(unnormalized_probs_t)
                unnormalized_probs_a = np.asarray(unnormalized_probs_a)
                if self.amount_biased != "amount_uniform":
                    unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, self.alpha)
                else:
                    unnormalized_probs = unnormalized_probs_t

                selected = self.weight_choice(unnormalized_probs)
                next_node = tmp_node[selected]
                next_time = tmp_time[selected]
                next_key = tmp_key[selected]
                return next_node, next_time, next_key

            else:
                return None, None, None  # 没有符合条件的


@njit(cache=True)
def get_first_step(cur_nbrs, nbr_keys, first_biased_type, max_time, min_time):
    tmp_key = []
    tmp_node = []
    tmp_time = []
    unnormalized_probs_t = []

    for i, nbr in enumerate(cur_nbrs):
        nbr_key = nbr_keys[i]  # cur领边的key数组
        for k in nbr_key:
            if k == -1:
                break
            t = k
            if first_biased_type == "time_uniform":
                unnormalized_probs_t.append(1)
            elif first_biased_type == "time_freq":
                unnormalized_probs_t.append(max_time - t + 1)
            elif first_biased_type == "time_close_linear":
                unnormalized_probs_t.append(max_time - t + 1)
            elif first_biased_type == "time_far":
                unnormalized_probs_t.append(t - min_time + 1)
            elif first_biased_type == "time_far_linear":
                unnormalized_probs_t.append(t)

            tmp_node.append(nbr)
            tmp_time.append(t)
            tmp_key.append(k)
    unnormalized_probs_t = np.array(unnormalized_probs_t)

    if first_biased_type == "time_close_linear":  # TBS descending
        unnormalized_probs_t = np.argsort(-unnormalized_probs_t) + 1
    elif first_biased_type == "time_far_linear":  # TBS ascending
        unnormalized_probs_t = np.argsort(unnormalized_probs_t) + 1

    if len(unnormalized_probs_t) > 0:  # 有符合条件的下一个点
        return unnormalized_probs_t, tmp_node, tmp_time, tmp_key

    return None, None, None, None


@njit(cache=True)
def get_next_step(cur_nbrs, nbr_keys, weight_keys, time_biased_type, amount_biased, max_time, alpha, prevtime=0):
    tmp_key = []
    tmp_node = []
    tmp_time = []
    unnormalized_probs_t = []
    unnormalized_probs_a = []

    for i, nbr in enumerate(cur_nbrs):
        nbr_key = nbr_keys[i]
        for j, k in enumerate(nbr_key):
            if k == -1:
                break
            t = k
            a = weight_keys[i][j]
            if time_biased_type == "no_time_limit":
                unnormalized_probs_t.append(1.0)

            elif t >= prevtime:
                unnormalized_probs_a.append(a)
                if time_biased_type == "time_uniform":
                    unnormalized_probs_t.append(1.0)
                elif time_biased_type == "time_close_raw":
                    unnormalized_probs_t.append(max_time - t + 1.0)
                elif time_biased_type == "time_close_exp":
                    unnormalized_probs_t.append(t - prevtime + 0.0)
                else:
                    unnormalized_probs_t.append(t - prevtime + 1.0)
                tmp_time.append(t)
                tmp_node.append(nbr)
                tmp_key.append(k)

    unnormalized_probs_t = np.array(unnormalized_probs_t, dtype=np.float64)
    unnormalized_probs_a = np.array(unnormalized_probs_a, dtype=np.float64)

    if time_biased_type == "time_close_linear":  # TBS descending
        unnormalized_probs_t = np.argsort(-unnormalized_probs_t) + 1.0
    elif time_biased_type == "time_far_linear":  # TBS ascending
        unnormalized_probs_t = np.argsort(unnormalized_probs_t) + 1.0

    if amount_biased == "amount_linear":  # WBS ascending
        unnormalized_probs_a = np.argsort(unnormalized_probs_a) + 1.0

    if len(unnormalized_probs_t) > 0:  # 有符合条件的下一个点
        if amount_biased != "amount_uniform":
            unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, alpha)
        else:
            unnormalized_probs = unnormalized_probs_t
        return unnormalized_probs, tmp_node, tmp_time, tmp_key

    return None, None, None, None


def get_tedge(args):
    path = args.features_file
    embeddings = pd.read_csv(path).values #8w6+
    sample_labels = load_labels('dataset/phishing/label.txt') # 8w6+ 编号的890个节点
    nodes = list([int(node) for node in sample_labels.keys()])
    phishing_nodes = nodes[:445]
    non_phishing_nodes = nodes[445:]
    nodes_labels = list(sample_labels.values())
    nodes_embeddings = pd.DataFrame(embeddings[nodes], index=nodes)
    X_train, X_test, y_train, y_test = train_test_split(nodes_embeddings, nodes_labels, train_size=args.train_size, random_state=args.seed)
    model = SVC(kernel='linear', C=0.4, random_state=args.seed)
    model.fit(X_train, y_train)

    return model.predict(embeddings)[args.nodes_to_keep]


def node_classification(args, output):
    if 'csv' not in output:
        output += '.csv'
    embeddings = pd.read_csv(output).values # 8w6+
    # labels = pd.read_csv('dataset/phishing/label.csv').values.ravel()
    sample_labels = load_labels('dataset/phishing/label.txt') # 890
    nodes = list([int(node) for node in sample_labels.keys()])
    nodes_labels = list(sample_labels.values())
    nodes_embeddings = pd.DataFrame(embeddings[nodes])

    X_train, X_test, y_train, y_test = train_test_split(nodes_embeddings, nodes_labels, train_size=args.train_size, random_state=args.seed)
    model = SVC(kernel='linear', C=0.4, random_state=args.seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    roc = roc_auc_score(y_test, y_pred)
    apc = average_precision_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print('roc_auc_score:{}'.format(roc))
    print('average_precision_score:{}'.format(apc))
    print('classification_report:\n{}'.format(cr))


def read_dataset(args):
    if args.dataset in ["tedge", "trans2vec"]:
        tG = tGraph('dataset/phishing/TransEdgelist.txt', verbose=args.verbose)
    elif "bc" in args.dataset:
        PUBLICDATA_PATH = os.path.abspath(os.path.expanduser("~/GraphData/datasets/")) + os.sep + "publicdata/"
        SAMPLE_GSIZE = int(args.dataset[2:]) * 10000
        SAMPLE_MULDIGS_PATH = os.path.join(PUBLICDATA_PATH, 'graph_%d/SP_MulDiGs.pkl' % SAMPLE_GSIZE)
        G = load_pickle(SAMPLE_MULDIGS_PATH)
        tG = tGraph(None, DiG=G, filetype="pd_f", verbose=args.verbose)
    else:
        assert "read_dataset invalid dataset:{}".format(args.dataset)
    return tG


def run_tedge(args):
    random_seed(args.seed)
    t1 = time.time()
    args.time_biased_type, args.first_biased_type, args.amount_biased, args.alpha = METHOD_MAP[args.tedge_type]
    output = args.outputdir + os.sep + "_".join([args.tedge_type, args.curtime, str(args.i)]) + '.csv'
    if args.run_emb == "true":
        tG = read_dataset(args)
        tGNE = tGraphNE(tG, args.time_biased_type, args.first_biased_type, args.amount_biased, args.alpha,
                        dimensions=args.dimensions, num_walks=args.num_walks,
                        walk_length=args.walk_length, window_size=args.window_size,
                        workers=args.workers, seed=args.seed, verbose=args.verbose, output=output,
                        save_features=True, is_dan=True, rac=False)
    else:
        print('skip embedding process...')

    if args.run_nc == "true":
        if args.run_emb == "false":
            assert len(args.filename) > 0, 'filename error'
            output = args.outputdir + os.sep + args.filename + '.csv'
        node_classification(args, output)
    else:
        print('skip node classification process...')
    print('run_tedge, cost:{} min'.format((time.time() - t1) / 60))


def get_tedge_model(args):
    random_seed(args.seed)
    args.time_biased_type, args.first_biased_type, args.amount_biased, args.alpha = METHOD_MAP[args.tedge_type]
    tG = read_dataset(args)
    tGNE = tGraphNE(tG, args.time_biased_type, args.first_biased_type, args.amount_biased, args.alpha,
                    dimensions=args.dimensions, num_walks=args.num_walks,
                    walk_length=args.walk_length, window_size=args.window_size,
                    workers=args.workers, seed=args.seed, verbose=args.verbose, output=None,
                    save_features=False, is_dan=True, rac=False)
    embeddings = tGNE.features
    sample_labels = load_labels('dataset/phishing/label.txt')  # 890
    nodes = list([int(node) for node in sample_labels.keys()])
    nodes_labels = list(sample_labels.values())
    nodes_embeddings = pd.DataFrame(embeddings[nodes])
    X_train, X_test, y_train, y_test = train_test_split(nodes_embeddings, nodes_labels, train_size=args.train_size,
                                                        random_state=args.seed)
    model = SVC(kernel='linear', C=0.4, random_state=args.seed)
    model.fit(X_train, y_train)
    y_pred = np.array(model.predict(X_test))
    acc = (np.array(y_test) == y_pred).mean()
    return tGNE, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-tt", "--tedge_type", default="TBS", choices=['TEDGE', 'TBS', 'WBS', 'TBS+WBS'], type=str)
    parser.add_argument("--run_emb", default="true", type=str)
    parser.add_argument("--run_nc", default="true", type=str)
    parser.add_argument("-f", "--filename", default="", type=str)
    parser.add_argument("-d", "--dimensions", default=128, type=int) # 128
    parser.add_argument("--num_walks", default=4, type=int) # 4
    parser.add_argument("--walk_length", default=10, type=int) # 10
    parser.add_argument("--window_size", default=4, type=int) # 4
    parser.add_argument("--workers", default=1, type=int) # 8
    parser.add_argument("--train_size", default=0.5, type=float)
    parser.add_argument("--rac", default="true", type=str, help="random_choice or alias table choice")
    args = parser.parse_args()
    args.rac = True if args.rac == "true" else False

    random_seed(args.seed)
    outputdir = "result/test_tedge"
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    args.outputdir = outputdir
    args.curtime = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    args.i = "test"
    run_tedge(args)

