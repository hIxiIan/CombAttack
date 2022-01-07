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
from cluster import make_redundancy


METHOD_MAP = {
    'TEDGE': ['time_uniform', 'time_uniform', 'amount_uniform', 1.0],
    'TBS': ['time_close_linear', 'time_close_linear', 'amount_uniform', 1.0],
    'WBS': ['time_uniform', 'time_uniform', 'amount_linear', 0.0],
    'TBS+WBS': ['time_close_linear', 'time_close_linear', 'amount_linear', 0.5],
}


def random_seed(seed=None):
    np.random.seed(seed)
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


def weight_choice(unnormalized_probs):
    norm_const = sum(unnormalized_probs)
    normalized_probs = np.array([float(u_prob / norm_const) for u_prob in unnormalized_probs])  # 归一化
    J = alias_setup(normalized_probs)[0]
    q = alias_setup(normalized_probs)[1]
    idx = alias_draw(J, q)
    return idx


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

    kk = int(np.floor(np.random.rand() * K))
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
        norm_const = np.sum(unnormalized_probs)
        normalized_probs = np.array([u_prob / norm_const for u_prob in unnormalized_probs])

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
    def __init__(self, file_, filetype='txt_f', output_Gpkl=False, verbose=0):
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
                 window_size=4, workers=1, hs=1, seed=2022, verbose=0, is_test_tedge_edges=False):
        self.G = tG.G
        self.min_time = tG.min_time
        self.max_time = tG.max_time
        self.verbose = verbose

        self.time_biased_type = time_biased_type  # choice = "unbiased", "amount-weighted" "linear", "exp"
        self.first_biased_type = first_biased_type
        self.amount_biased = amount_biased
        self.alpha = alpha
        t1 = time.time()
        walks = self.simulate_walks(num_walks, walk_length)  # 随机游走
        t2 = time.time()
        word2vec_model = Word2Vec(sentences=walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, hs=1,
                                  workers=workers, seed=seed)
        t3 = time.time()
        vectors = word2vec_model.wv.vectors
        index_to_key = np.array(word2vec_model.wv.index_to_key).astype(int)
        tup = sorted(zip(vectors, index_to_key), key=lambda x: x[1], reverse=False)
        features = np.array([t[0] for t in tup])
        if not is_test_tedge_edges:
            pd.DataFrame(features).to_csv(output, index=None)
        else:
            self.features = features

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
            random.shuffle(nodes)
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
        weight_keys = []
        nbr_keys = []
        for nbr in cur_nbrs:
            nbr_key = list(G.get_edge_data(cur, nbr))  # cur领边的key数组
            nbr_keys.append(nbr_key)
            weight_keys.append([G[cur][nbr][nk]['weight'] for nk in nbr_key])
        nbr_keys = np.array(make_redundancy(nbr_keys), dtype=np.int32)
        weight_keys = np.array(make_redundancy(weight_keys), dtype=np.float32)

        unnormalized_probs_t, tmp_node, tmp_time, tmp_key = get_first_step(cur_nbrs, nbr_keys, self.first_biased_type, self.max_time, self.min_time)
        selected = weight_choice(unnormalized_probs_t)
        next_node = tmp_node[selected]
        next_time = tmp_time[selected]
        next_key = tmp_key[selected]
        if next_node is not None:
            walk.append(next_node)
            walk_time.append(next_time)
            walk_edge.append(next_key)
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
            unnormalized_probs_t, tmp_node, tmp_time, tmp_key = self.get_next_step(cur_nbrs, nbr_keys, weight_keys, self.time_biased_type, self.amount_biased, self.max_time, self.max_time, prevtime)
            selected = weight_choice(unnormalized_probs_t)
            next_node = tmp_node[selected]
            next_time = tmp_time[selected]
            next_key = tmp_key[selected]
            if next_node is not None:
                walk.append(next_node)
                walk_time.append(next_time)
                walk_edge.append(next_key)
            else:
                break
        return walk


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
    path = args.tedge_features_file
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


def run_tedge(args):
    t1 = time.time()
    args.time_biased_type, args.first_biased_type, args.amount_biased, args.alpha = METHOD_MAP[args.tedge_type]
    output = args.outputdir + os.sep + "_".join([args.tedge_type, args.curtime, str(args.i)]) + '.csv'
    if args.run_emb == "true":
        tG = tGraph('dataset/phishing/TransEdgelist.txt', verbose=args.verbose)
        tGNE = tGraphNE(tG, args.time_biased_type, args.first_biased_type, args.amount_biased, args.alpha,
                        dimensions=args.dimensions, num_walks=args.num_walks,
                        walk_length=args.walk_length, window_size=args.window_size,
                        workers=args.workers, seed=args.seed, verbose=args.verbose, output=output)
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
    args = parser.parse_args()

    random_seed(args.seed)
    outputdir = "result/test_tedge"
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    args.outputdir = outputdir
    args.curtime = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    args.i = "test"
    run_tedge(args)

