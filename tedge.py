import sys
import networkx as nx
import pickle
import numpy as np
import time
import random
import argparse
import torch
import os
from gensim.models import Word2Vec
from time import strftime, localtime
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import train_test_split


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
    if len(unnormalized_probs) > 0:  # 有符合条件的下一个点
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob / norm_const) for u_prob in unnormalized_probs]  # 归一化

        J = alias_setup(normalized_probs)[0]
        q = alias_setup(normalized_probs)[1]
        idx = alias_draw(J, q)
    return idx


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


def normalized_probs(unnormalized_probs):
    if len(unnormalized_probs) > 0:  # 有符合条件的下一个点
        norm_const = sum(unnormalized_probs)
        normalized_probs = [u_prob / norm_const for u_prob in unnormalized_probs]  # 归一化

    return normalized_probs


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

        if filetype == "txt_raw":
            with open(file_) as f:
                for l in f:
                    x, y = l.strip().split(' ')
                    self.G.add_edge(x, y, key=edge_key)
                    edge_key = edge_key + 1
        else:
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

            elif filetype == "pkl_f":
                with open( file_ ,"rb") as f:
                    df_in = pickle.load(f)
                for i in df_in.index:
                    x = str(int(df_in.From[i]))
                    y = str(int(df_in.To[i]))
                    t = int(df_in.TimeStamp[i])
                    a = df_in.Value[i]

                    if self.G.has_edge(x ,y ,t):
                        self.G[x][y][t]['weight'] += a
                    else:
                        self.G.add_edge(x ,y ,key=t, weight=a)

                    edge_key = edge_key + 1
                    if t < self.min_time:
                        self.min_time = t
                    elif t > self.max_time:
                        self.max_time = t

            if output_Gpkl == True:
                pklfile_G = "tGraph.pickle"
                with open(pklfile_G, "wb") as f:
                    print("Writing", pklfile_G, "...")
                    pickle.dump( self.G, pklfile_G, pickle.HIGHEST_PROTOCOL )

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
    def __init__(self, tG, time_biased_type, first_biased_type, amount_biased, alpha,
                 dimensions, num_walks, walk_length, output, output_pklG=False,
                 window_size=10, workers=8, hs=1, seed=2022, verbose=0):
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
        # walks = [map(str, walk) for walk in walks]
        word2vec_model = Word2Vec(sentences=walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, hs=1,
                                  workers=workers, seed=seed)
        t3 = time.time()
        self.vectors = {}
        for word in list(self.G.nodes()):
            self.vectors[str(word)] = word2vec_model.wv[str(word)]
        word2vec_model.wv.save_word2vec_format(output)
        # self.word2vec_model = word2vec_model
        del word2vec_model

        if verbose > 0:
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
                selected = weight_choice(unnormalized_probs_t)
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
                if self.amount_biased != "amount_uniform":
                    unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, self.alpha)
                else:
                    unnormalized_probs = unnormalized_probs_t

                selected = weight_choice(unnormalized_probs)
                next_node = tmp_node[selected]
                next_time = tmp_time[selected]
                next_key = tmp_key[selected]
                return next_node, next_time, next_key

            else:
                return None, None, None  # 没有符合条件的


def node_classification(args, output):
    embeddings = load_embeddings(output)
    labels = load_labels('dataset/phishing/label.txt')
    nodes = list(labels.keys())
    nodes_labels = list(labels.values())
    nodes_embeddings = np.array([embeddings[node] for node in nodes])

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


def get_tedge_dataset(args):
    file_dir = 'dataset/phishing/'
    edges_txt = 'TransEdgelist.txt'
    labels_txt = 'label.txt'
    tG = tGraph(file_dir + edges_txt, verbose=args.verbose)
    labels = load_labels(file_dir + labels_txt)
    args.tedge_tG = tG
    args.tedge_labels = labels


def run_tedge(args):
    t1 = time.time()
    args.time_biased_type, args.first_biased_type, args.amount_biased, args.alpha = METHOD_MAP[args.tedge_type]
    output = args.outputdir + os.sep + "_".join([args.tedge_type, args.curtime, str(args.i)])
    if args.run_emb == "true":
        get_tedge_dataset(args)
        tGNE = tGraphNE(args.tedge_tG, args.time_biased_type, args.first_biased_type, args.amount_biased, args.alpha,
                        dimensions=args.dimensions, num_walks=args.num_walks,
                        walk_length=args.walk_length, window_size=args.window_size,
                        workers=args.workers, seed=args.seed, verbose=args.verbose, output=output)
    else:
        print('skip embedding process...')

    if args.run_nc == "true":
        if args.run_emb == "false":
            assert len(args.filename) > 0, 'filename error'
            output = args.outputdir + os.sep + args.filename
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
    parser.add_argument("--run_emb", default="false", type=str)
    parser.add_argument("--run_nc", default="true", type=str)
    parser.add_argument("-f", "--filename", default="TBS_2022_01_04_13_58_21", type=str)
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
    args.i = ""
    run_tedge(args)

