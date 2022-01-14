import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from gensim.models import Word2Vec
from time import strftime, localtime
from sklearn.svm import SVC, OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import train_test_split
from numba import jit, njit
from tGraph import tGraph
from walker import BiasedRandomWalker, BiasedRandomWalkerAlias
from time import time


@njit
def numba_seed(sd):
    np.random.seed(sd)


def random_seed(seed=None):
    np.random.seed(seed)
    numba_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@jit(cache=True, nopython=True)
def weight_choice(unnormalized_probs):
    norm_const = np.sum(unnormalized_probs)
    normalized_probs = np.array([float(u_prob / norm_const) for u_prob in unnormalized_probs])  # 归一化
    J, q = alias_setup(normalized_probs)
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
    kk = np.random.randint(K)
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


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


@jit(cache=True, nopython=True)
def normalized_probs(unnormalized_probs):
    if len(unnormalized_probs) > 0:  # 有符合条件的下一个点
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()
    return normalized_probs


@jit(cache=True, nopython=True)
def combine_probs(p1, p2, alpha):
    probs1 = normalized_probs(p1)
    probs2 = normalized_probs(p2)

    assert len(probs1) == len(probs2), "combine_probs invalid"
    combine_probs = np.multiply(np.power(probs1, alpha), np.power(probs2, 1 - alpha))
    return combine_probs


class trans2vec(object):
    def __init__(self, output, perturbed_tuple=None, alpha=0.5, gf_alias_mode=True,
                 dimensions=64, num_walks=20, walk_length=5,
                 window_size=10, workers=1, hs=1, seed=2022, verbose=0, save_features=False, gf=True):
        self.perturbed_tuple = perturbed_tuple
        self.verbose = verbose
        self.alpha = alpha
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.output = output
        self.seed = seed
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.save_features = save_features
        self.gf_alias_mode = gf_alias_mode

        self.walks = None
        self.word2vec_model = None
        self.features = None
        self.do(gf=gf)

    def do(self, gf=True):
        if gf:
            if self.verbose > 0:
                print('run gf w2v')
            self.gf_walk()
            return
        if self.verbose > 0:
            print('run dan w2v')
        self.dan_walk()

    # gf_walk 预处理转移概率 alpha * TBS * (1-alpha) * WBS
    def get_amount_timestamp_data(self):
        N = self.adj_matrix.shape[0]
        amount_timestamp_data = sp.lil_matrix((N, N), dtype=np.float64)
        nodes = np.arange(N, dtype=np.int32)
        indices = self.adj_matrix.indices
        indptr = self.adj_matrix.indptr
        amount_data = self.amount_data.data
        timestamp_data = self.timestamp_data.data
        for node in nodes:
            nbrs = indices[indptr[node]: indptr[node + 1]]
            nbrs_amount_probs = amount_data[indptr[node]: indptr[node + 1]].copy()
            nbrs_timestamp_probs = timestamp_data[indptr[node]: indptr[node + 1]].copy()
            nbrs_unnormalized_probs = combine_probs(nbrs_amount_probs, nbrs_timestamp_probs, self.alpha)

            for i, nbr in enumerate(nbrs):
                amount_timestamp_data[node, nbr] = nbrs_unnormalized_probs[i]
        amount_timestamp_data = amount_timestamp_data.tocsr()
        return amount_timestamp_data.data

    def gf_walk(self):
        if self.perturbed_tuple is None:
            data = np.load('dataset/phishing/trans2vec.npz', allow_pickle=True)
            self.adj_matrix = data['adj_matrix'].item()
            self.amount_data = data['amount_data'].item()
            self.timestamp_data = data['timestamp_data'].item()
            self.node_label = data['node_label']
            self.adj_matrix.data = self.get_amount_timestamp_data()
        else:
            self.adj_matrix = self.perturbed_tuple[0]
            self.amount_data = self.perturbed_tuple[1]
            self.timestamp_data = self.perturbed_tuple[2]
            self.adj_matrix.data = self.get_amount_timestamp_data()
        t1 = time()
        if self.gf_alias_mode:
            if self.verbose > 0:
                print("run gf alias mode")
            walks = BiasedRandomWalkerAlias(walk_length=self.walk_length, walk_number=self.num_walks,
                                        p=1.0, q=1.0, extend=False, mode="SparseOTF").walk(self.adj_matrix)
        else:
            if self.verbose > 0:
                print("run gf normal mode")
            walks = BiasedRandomWalker(walk_length=self.walk_length, walk_number=self.num_walks).walk(self.adj_matrix)
        t2 = time()
        word2vec_model = Word2Vec(sentences=walks, vector_size=self.dimensions, window=self.window_size,
                                  min_count=0, sg=1, hs=1, workers=self.workers, seed=self.seed)
        t3 = time()
        features = word2vec_model.wv.vectors[np.fromiter(map(int, word2vec_model.wv.index_to_key), np.int32).argsort()]

        if self.save_features:
            pd.DataFrame(features).to_csv(self.output, index=None)

        self.walks = walks
        self.word2vec_model = word2vec_model
        self.features = features

        if self.verbose > 0:
            print("walk cost: {} min".format((t2 - t1) / 60))
            print("w2v cost: {} min".format((t3 - t2) / 60))

    def dan_walk(self):
        tG = tGraph(verbose=self.verbose)
        self.G = tG.G
        t1 = time()
        walks = self.simulate_walks(self.num_walks, self.walk_length)
        t2 = time()
        word2vec_model = Word2Vec(sentences=walks, vector_size=self.dimensions, window=self.window_size,
                                  min_count=0, sg=1, hs=1, workers=self.workers, seed=self.seed)
        t3 = time()
        features = word2vec_model.wv.vectors[np.fromiter(map(int, word2vec_model.wv.index_to_key), np.int32).argsort()]

        if self.save_features:
            pd.DataFrame(features).to_csv(self.output, index=None)

        self.walks = walks
        self.word2vec_model = word2vec_model
        self.features = features

        if self.verbose > 0:
            print("walk cost: {} min".format((t2 - t1) / 60))
            print("w2v cost: {} min".format((t3 - t2) / 60))

    def simulate_walks(self, num_walks, walk_length):
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
        walk = [start_node]
        walk_edge = []
        walk_time = []

        cur = start_node
        next_node, next_time, next_key = self.get_next_step(cur)
        if next_node is not None:
            walk.append(next_node)
            walk_time.append(next_time)
            walk_edge.append(next_key)
        else:
            return walk

        while len(walk) < walk_length:
            cur = walk[-1]
            next_node, next_time, next_key = self.get_next_step(cur)
            if next_node is not None:
                walk.append(next_node)
                walk_time.append(next_time)
                walk_edge.append(next_key)
            else:
                break
        return walk

    def get_next_step(self, cur):
        G = self.G
        tmp_key = []
        tmp_node = []
        tmp_time = []
        unnormalized_probs_t = []
        unnormalized_probs_a = []

        cur_nbrs = list(G.neighbors(cur))
        for nbr in cur_nbrs:
            nbr_key = list(G.get_edge_data(cur, nbr))
            k = nbr_key[-1]
            t = k
            a = G[cur][nbr][k]['weight']
            unnormalized_probs_a.append(a)
            unnormalized_probs_t.append(len(nbr_key))
            tmp_time.append(t)
            tmp_node.append(nbr)
            tmp_key.append(k)

        if len(unnormalized_probs_t) > 0:  # 有符合条件的下一个点
            unnormalized_probs_t = np.asarray(unnormalized_probs_t)
            unnormalized_probs_a = np.asarray(unnormalized_probs_a)
            unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, self.alpha)
            selected = weight_choice(unnormalized_probs)
            next_node = tmp_node[selected]
            next_time = tmp_time[selected]
            next_key = tmp_key[selected]
            return next_node, next_time, next_key
        else:
            return None, None, None  # 没有符合条件的


# trans2vec原论文的数据集实验设置不明，暂时没有复现原来的效果
def node_classification(args, output):
    if 'csv' not in output:
        output += '.csv'
    embeddings = pd.read_csv(output).values # 8w6+
    # labels = pd.read_csv('dataset/phishing/label.csv').values.ravel()
    sample_labels = load_labels('dataset/phishing/label.txt') # 890
    nodes = list([int(node) for node in sample_labels.keys()])
    nodes_labels = list(sample_labels.values())
    nodes_embeddings = pd.DataFrame(embeddings[nodes])

    model_name = args.trans2vec_model.lower()

    if model_name in ["ocsvm"]:
        # 训练集是无标签节点，测试集是标签+无标签节点
        # X_train, X_test, y_test = nodes_embeddings[445:], nodes_embeddings, nodes_labels

        # 训练集是钓鱼节点，测试集是钓鱼与非钓鱼节点
        # X_train, X_test, y_test = nodes_embeddings[:445], nodes_embeddings, nodes_labels

        # 总数据随机选80%
        # X_train, X_test, y_train, y_test = train_test_split(nodes_embeddings, nodes_labels, train_size=args.train_size, random_state=args.seed)

        # 分别随机选80%，分层抽样
        # X_train, X_test, y_train, y_test = train_test_split(nodes_embeddings, nodes_labels, train_size=args.train_size, random_state=args.seed, stratify=nodes_labels)

        # 有标签数据的80%作为训练集，20%作为测试集
        X_train, X_test, y_train, y_test = train_test_split(nodes_embeddings[:445], nodes_labels[:445], train_size=args.train_size, random_state=args.seed)
        model = OneClassSVM(nu=0.05, gamma="auto", kernel="rbf", tol=1e-3).fit(X_train)
        y_pred = model.predict(X_test)
        y_pred = np.array([1 if _y == 1 else 0 for _y in y_pred])
        # acc = np.mean(y_pred == y_test) 就是cr中1的precision
        cr = classification_report(y_pred, y_test)
    elif model_name in ["svm", 'lr']:
        X_train, X_test, y_train, y_test = train_test_split(nodes_embeddings, nodes_labels, train_size=args.train_size, random_state=args.seed)
        if model_name == "svm":
            model = SVC(kernel='linear', C=0.4, random_state=args.seed)
        elif model_name == "lr":
            model = LogisticRegression(random_state=args.seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cr = classification_report(y_pred, y_test)
    else:
        assert False, "get_model invalid model"
    print('classification_report:\n{}'.format(cr))


def run_trans2vec(args):
    t1 = time()
    output = args.outputdir + os.sep + "_".join([args.tedge_type, args.curtime, str(args.i)]) + '.csv'
    if args.run_emb == "true":
        t2v = trans2vec(output, alpha=args.alpha, dimensions=args.dimensions, num_walks=args.num_walks,
                        walk_length=args.walk_length, window_size=args.window_size, save_features=True,
                        workers=args.workers, seed=args.seed, verbose=args.verbose, gf_alias_mode=args.gf_alias_mode, gf=args.gf)
    else:
        print('skip embedding process...')

    if args.run_nc == "true":
        if args.run_emb == "false":
            assert len(args.filename) > 0, 'filename error'
            output = args.outputdir + os.sep + args.filename + '.csv'
        node_classification(args, output)
    else:
        print('skip node classification process...')
    print('run_trans2vec, cost:{} min'.format((time() - t1) / 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=1, type=int, help="print details if verbose > 0")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-tt", "--tedge_type", default="TBS+WBS", choices=['TBS+WBS'], type=str, help="trans2vec settings")
    parser.add_argument("--run_emb", default="true", type=str, help="run embedding process")
    parser.add_argument("--run_nc", default="true", type=str, help="run node classification process")
    parser.add_argument("-f", "--filename", default="TBS+WBS_2022_01_12_18_31_57_test", type=str, help="saved embedding filepath")
    parser.add_argument("-d", "--dimensions", default=64, type=int, help="random walk parameters") # 128
    parser.add_argument("--num_walks", default=20, type=int, help="random walk parameters")
    parser.add_argument("--walk_length", default=5, type=int, help="random walk parameters")
    parser.add_argument("--window_size", default=10, type=int, help="random walk parameters")
    parser.add_argument("--workers", default=1, type=int, help="random walk parameters")
    parser.add_argument("--train_size", default=0.8, type=float, help="node classification task train ratio")
    parser.add_argument("--trans2vec_model", default="ocsvm", type=str, help="node classification machine learning model")
    parser.add_argument("--alpha", default=0.5, type=float, help="the parameter of balance between TBS and WBS")
    parser.add_argument("--gf_alias_mode", default="false", type=str)
    parser.add_argument("--gf", default="true", type=str)
    args = parser.parse_args()
    args.gf = True if args.gf == "true" else False
    args.gf_alias_mode = True if args.gf and args.gf_alias_mode == "true" else False

    random_seed(args.seed)
    outputdir = "result/test_trans2vec"
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    args.outputdir = outputdir
    args.curtime = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    args.i = "test"
    run_trans2vec(args)

