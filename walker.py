import random
import numpy as np
from graphgallery import functional as gf
from utils import get_purity, stochastic_accept, get_wl, get_hop_neighbors, get_cross_entropy_target_nbrs, get_cross_entropy_list, get_wl_list, get_purity_list, get_purity_gains, get_wl_gains, random_choice
from time import time
from numba import jit, int64, int32
from sklearn import preprocessing


class Walker:
    def __init__(self, targets, sample_ratio, subgraph_type, adj_matrix, labels, p=1.0, q=1.0, ori_logits=None, logits=None, wl_limit=1.0, eps=1e-4):
        self.p = p
        self.q = q
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.adj_matrix = adj_matrix.tolil()  # weight默认为0/1
        self.adj_matrix_csr = adj_matrix
        self.labels = labels
        self.subgraph_type = subgraph_type
        self.ori_logits = ori_logits
        self.logits = logits

        self.purity = None
        self.purity_r = None
        self.wl_limit = wl_limit
        self.purity_list = None
        self.ce_list = None
        self.wl_list = None
        self.eps = eps
        self.min_max_scaler = preprocessing.MinMaxScaler()

        self.wrong_labels = None
        self.targets_map = {}
        self.sample_edges = []
        self.sample_nodes = []
        self.targets = np.asarray(targets)
        for i, target in enumerate(self.targets):
            self.targets_map[target] = i
        self.sample_nums = int(sample_ratio * adj_matrix.shape[0])
        self.wls = None
        self.wl_cnts = None
        self.alias_edges_ = []
        self.alias_nodes_ = []

    def get_wrong_labels(self):
        wrong_labels = []
        for target in self.targets:
            logit = self.ori_logits[target]
            idx = list(set(range(logit.size)) - set([self.labels[target]]))
            wrong_label = idx[logit[idx].argmax()]
            wrong_labels.append(wrong_label)
        self.wrong_labels = np.array(wrong_labels)

    def get_all_wl(self):
        self.get_wrong_labels()
        wls = []
        wl_cnts = []
        for wrong_label in self.wrong_labels:
            wl, wl_cnt = get_wl(self.indices, self.indptr, self.labels, wrong_label,
                                          self.eps)
            wls.append(wl)
            wl_cnts.append(wl_cnt)

            if 'n2v' in self.subgraph_type:
                self.wl_list = get_wl_list(self.indices, self.indptr, wl)
                self.preprocess_transition_probs()
                self.alias_edges_.append(self.alias_edges)
                self.alias_nodes_.append(self.alias_nodes)

        self.wls = np.array(wls)
        self.wl_cnts = np.array(wl_cnts)

    def get_all_purity(self):
        self.get_wrong_labels()
        self.purity = get_purity(self.indices, self.indptr, self.labels)  # 纯度
        self.purity_r = 1 - self.purity + self.eps  # 杂度
        self.purity += self.eps

        if 'n2v' in self.subgraph_type:
            self.purity_list = get_purity_list(self.indices, self.indptr, self.purity_r)
            self.preprocess_transition_probs()

    def get_all_ce(self):
        if 'n2v' not in self.subgraph_type:
            return
        self.ce_list = get_cross_entropy_list(self.indices, self.indptr, self.logits)
        self.preprocess_transition_probs()

    def random_walk(self):
        if 'dw' in self.subgraph_type:
            if 'wl' in self.subgraph_type:
                self.get_all_wl()
                if self.subgraph_type == 'dw_wl':
                    sub_edges, sub_nodes = self.deepwalk_wl_sample(self.wls, self.indices, self.indptr, self.targets, self.sample_nums, is_topk=False)
                elif self.subgraph_type == 'dw_wl_topk':
                    sub_edges, sub_nodes = self.deepwalk_wl_sample(self.wls, self.indices, self.indptr, self.targets, self.sample_nums, is_topk=True)
                elif self.subgraph_type == 'dw_wl_dynamic':
                    sub_edges, sub_nodes = self.deepwalk_wl_dynamic_sample(self.wls, self.indices, self.indptr, self.targets, self.sample_nums, is_topk=False)
                elif self.subgraph_type == 'dw_wl_dynamic_topk':
                    sub_edges, sub_nodes = self.deepwalk_wl_dynamic_sample(self.wls, self.indices, self.indptr, self.targets, self.sample_nums, is_topk=True)
                elif self.subgraph_type == 'dw_wl_gains':
                    sub_edges, sub_nodes = self.deepwalk_wl_gains_sample(self.wls, self.labels, self.wrong_labels, self.indices, self.indptr, self.targets, self.sample_nums)
                elif self.subgraph_type == 'dw_wl_kh':
                    sub_edges, sub_nodes = self.deepwalk_wl_kh_sample(self.targets, self.sample_nums)
                elif self.subgraph_type == 'dw_biased_wl':
                    sub_edges, sub_nodes = self.deepwalk_biased_wl_sample(self.wls, self.p, self.q, self.indices, self.indptr, self.targets, self.sample_nums)
            elif 'purity' in self.subgraph_type:
                self.get_all_purity()
                if self.subgraph_type == 'dw_purity':
                    sub_edges, sub_nodes = self.deepwalk_purity_sample(self.purity, self.purity_r, self.labels, self.wrong_labels, self.indices, self.indptr, self.targets, self.sample_nums, is_topk=False)
                elif self.subgraph_type == 'dw_purity_topk':
                    sub_edges, sub_nodes = self.deepwalk_purity_sample(self.purity, self.purity_r, self.labels, self.wrong_labels, self.indices, self.indptr, self.targets, self.sample_nums, is_topk=True)
                elif self.subgraph_type == 'dw_purity_gains':
                    sub_edges, sub_nodes = self.deepwalk_purity_gains_sample(self.purity, self.labels, self.indices, self.indptr, self.targets, self.sample_nums)
                elif self.subgraph_type == 'dw_purity_gains_select':
                    sub_edges, sub_nodes = self.deepwalk_purity_gains_select_sample(self.targets, self.sample_nums, is_topk=False)
                elif self.subgraph_type == 'dw_purity_gains_select_topk':
                    sub_edges, sub_nodes = self.deepwalk_purity_gains_select_sample(self.targets, self.sample_nums, is_topk=True)
                elif self.subgraph_type == 'dw_biased_purity':
                    sub_edges, sub_nodes = self.deepwalk_biased_purity_sample(self.purity, self.purity_r, self.labels, self.wrong_labels, self.p, self.q, self.indices, self.indptr, self.targets, self.sample_nums)
            elif 'ce' in self.subgraph_type:
                if self.subgraph_type == 'dw_ce':
                    sub_edges, sub_nodes = self.deepwalk_ce_sample(self.logits, self.indices,self.indptr, self.targets, self.sample_nums, is_topk=False)
                elif self.subgraph_type == 'dw_ce_topk':
                    sub_edges, sub_nodes = self.deepwalk_ce_sample(self.logits, self.indices,self.indptr, self.targets, self.sample_nums, is_topk=True)
                elif self.subgraph_type == 'dw_ce_dynamic':
                    sub_edges, sub_nodes = self.deepwalk_ce_dynamic_sample(self.logits, self.indices,self.indptr, self.targets, self.sample_nums, is_topk=False)
                elif self.subgraph_type == 'dw_ce_dynamic_topk':
                    sub_edges, sub_nodes = self.deepwalk_ce_dynamic_sample(self.logits, self.indices,self.indptr, self.targets, self.sample_nums, is_topk=True)
            else:
                if self.subgraph_type == 'dw':
                    sub_edges, sub_nodes = self.deepwalk_sample(self.indices, self.indptr, self.targets, self.sample_nums)
                elif self.subgraph_type == 'dw_biased':
                    sub_edges, sub_nodes = self.deepwalk_biased_sample(self.p, self.q, self.indices, self.indptr, self.targets, self.sample_nums)
        elif 'n2v' in self.subgraph_type:
            if 'wl' in self.subgraph_type:
                self.get_all_wl()
            elif 'purity' in self.subgraph_type:
                self.get_all_purity()
            elif 'ce' in self.subgraph_type:
                self.get_all_ce()
            else:
                self.preprocess_transition_probs()
            sub_edges, sub_nodes = self.node2vec_sample(self.targets, self.sample_nums)

        self.sample_edges = [gf.asedge(sub_edge, shape='row_wise') if len(sub_edge) > 0 else np.array([[], []], dtype='int64') for sub_edge in sub_edges]
        self.sample_nodes = [np.unique(sub_node) for sub_node in sub_nodes]

    # 纯随机游走
    @staticmethod
    @jit(cache=True, nopython=True, locals={'head': int64, 'u': int64})
    def deepwalk_sample(indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        for target in targets:
            tmp_edges = {}
            tmp_nodes = [target]
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = indices[indptr[head]:indptr[head + 1]]

                if len(nbrs) > 0:
                    u = np.random.choice(nbrs)
                    tmp_nodes.append(u)
                    if (u, head) not in tmp_edges:
                        tmp_edges[(head, u)] = 1
                else:
                    break
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    @staticmethod
    @jit(cache=True, nopython=True, locals={'current_node': int64, 'u': int64})
    def deepwalk_biased_sample(p, q, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        N = len(indptr) - 1
        for target in targets:
            tmp_edges = {}
            tmp_nodes = [target]
            current_node = target
            previous_node = N
            previous_node_neighbors = np.empty(0, dtype=np.int32)
            while len(tmp_nodes) < sample_nums:
                neighbors = indices[indptr[current_node]:indptr[current_node + 1]]
                if neighbors.size == 0:
                    break

                probability = np.array([1 / q] * neighbors.size)
                probability[previous_node == neighbors] = 1 / p
                for i, nbr in enumerate(neighbors):
                    if np.any(nbr == previous_node_neighbors):
                        probability[i] = 1.

                norm_probability = probability / np.sum(probability)
                u = random_choice(neighbors, norm_probability)
                tmp_nodes.append(u)
                if (u, current_node) not in tmp_edges:
                    tmp_edges[(current_node, u)] = 1
                previous_node_neighbors = neighbors
                previous_node = u
                current_node = u
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    @staticmethod
    @jit(cache=True, nopython=True, locals={'current_node': int64, 'u': int64})
    def deepwalk_biased_wl_sample(wls, p, q, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        N = len(indptr) - 1
        for i, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            current_node = target
            previous_node = N
            previous_node_neighbors = np.empty(0, dtype=np.int32)
            while len(tmp_nodes) < sample_nums:
                neighbors = indices[indptr[current_node]:indptr[current_node + 1]]
                if neighbors.size == 0:
                    break

                nbrs_wl = wls[i][neighbors]
                probability = np.array([1 / q] * neighbors.size)
                probability[previous_node == neighbors] = 1 / p
                for i, nbr in enumerate(neighbors):
                    if np.any(nbr == previous_node_neighbors):
                        probability[i] = 1.
                probability = probability * nbrs_wl
                norm_probability = probability / np.sum(probability)
                # norm_probability = norm_probability * nbrs_wl
                # norm_probability = norm_probability / np.sum(norm_probability)
                u = random_choice(neighbors, norm_probability)
                tmp_nodes.append(u)
                if (u, current_node) not in tmp_edges:
                    tmp_edges[(current_node, u)] = 1
                previous_node_neighbors = neighbors
                previous_node = u
                current_node = u
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    @staticmethod
    @jit(cache=True, nopython=True, locals={'head': int64, 'u': int64})
    def deepwalk_wl_sample(wls, indices, indptr, targets, sample_nums, is_topk):
        edges = []
        nodes = []

        for i, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            topk = len(indices[indptr[target]:indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = indices[indptr[head]:indptr[head + 1]]

                if len(nbrs) == 0:
                    break
                nbrs_wl = wls[i][nbrs]
                if is_topk and topk < len(nbrs):
                    idx_topk = np.argsort(nbrs_wl)[-topk:]
                    nbrs = nbrs[idx_topk]
                    nbrs_wl = nbrs_wl[idx_topk]

                u = nbrs[stochastic_accept(nbrs_wl)]
                tmp_nodes.append(u)
                if (u, head) not in tmp_edges:
                    tmp_edges[(head, u)] = 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    # 轮盘赌+动态wrong_label阈值
    # 轮盘赌+动态wrong_label阈值+topk
    @staticmethod
    @jit(cache=True, nopython=True, locals={'head': int64, 'u': int64})
    def deepwalk_wl_dynamic_sample(wls, indices, indptr, targets, sample_nums, is_topk):
        edges = []
        nodes = []

        for i, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            wl_limit = 0.5
            topk = len(indices[indptr[target]:indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = indices[indptr[head]:indptr[head + 1]]
                if len(nbrs) == 0:
                    break
                nbrs_wl = wls[i][nbrs]
                one_wl_idx = nbrs_wl >= wl_limit
                if one_wl_idx.sum() >= topk:
                    nbrs = nbrs[one_wl_idx]
                    nbrs_wl = wls[i][nbrs]

                if is_topk and topk < len(nbrs):
                    idx_topk = np.argsort(nbrs_wl)[-topk:]
                    nbrs = nbrs[idx_topk]
                    nbrs_wl = nbrs_wl[idx_topk]

                u = nbrs[stochastic_accept(nbrs_wl)]
                tmp_nodes.append(u)
                wl_limit = max(wl_limit, wls[i][u])
                if (u, head) not in tmp_edges:
                    tmp_edges[(head, u)] = 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    @staticmethod
    @jit(cache=True, nopython=True, locals={'head': int32, 'u': int32})
    def deepwalk_wl_gains_sample(wls, labels, wrong_labels, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        left = 1
        for i, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            while len(tmp_nodes) < sample_nums:
                head = np.random.choice(np.array(tmp_nodes[-left:]))
                nbrs = indices[indptr[head]:indptr[head + 1]]

                if len(nbrs) == 0:
                    break
                nbrs_wl_gains = get_wl_gains(indices, indptr, labels, wls[i], target,
                                             nbrs, wrong_labels[i])

                nbrs_wl_gains = nbrs_wl_gains.ravel()
                idx = nbrs_wl_gains <= 0
                if idx.sum() > 0:
                    nbrs = nbrs[idx]
                    left = len(nbrs)
                    for u in nbrs:
                        tmp_nodes.append(u)
                        if (u, head) not in tmp_edges:
                            tmp_edges[(head, u)] = 1
                else:
                    # nbrs_wl_gains = self.min_max_scaler.fit_transform(nbrs_wl_gains).ravel()
                    # nbrs_wl_gains = nbrs_wl_gains + self.eps
                    # u = nbrs[stochastic_accept(nbrs_wl_gains)]
                    nbrs_wl_gains = nbrs_wl_gains[~idx]
                    nbrs = nbrs[~idx]
                    u = nbrs[nbrs_wl_gains.argmax()]
                    # u = random.choice(nbrs)
                    left = 1
                    tmp_nodes.append(u)
                    if (u, head) not in tmp_edges:
                        tmp_edges[(head, u)] = 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    # 轮盘赌
    # 轮盘赌+topk
    @staticmethod
    @jit(cache=True, nopython=True, locals={'head': int64, 'u': int64})
    def deepwalk_ce_sample(logits, indices, indptr, targets, sample_nums, is_topk):
        edges = []
        nodes = []

        for target in targets:
            tmp_edges = {}
            tmp_nodes = [target]
            topk = len(indices[indptr[target]:indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = indices[indptr[head]:indptr[head + 1]]
                nbrs_ce = get_cross_entropy_target_nbrs(target, nbrs, logits)
                if len(nbrs) == 0:
                    break
                if is_topk and topk < len(nbrs):
                    idx_topk = np.argsort(nbrs_ce)[-topk:]
                    nbrs = nbrs[idx_topk]
                    nbrs_ce = nbrs_ce[idx_topk]

                u = nbrs[stochastic_accept(nbrs_ce)]
                tmp_nodes.append(u)
                if (u, head) not in tmp_edges:
                    tmp_edges[(head, u)] = 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    # 轮盘赌+动态cross_entropy阈值
    # 轮盘赌+动态cross_entropy阈值+topk
    @staticmethod
    @jit(cache=True, nopython=True, locals={'head': int64, 'u': int64})
    def deepwalk_ce_dynamic_sample(logits, indices, indptr, targets, sample_nums, is_topk):
        edges = []
        nodes = []

        for target in targets:
            tmp_edges = {}
            tmp_nodes = [target]
            ce_limit = 1.0
            topk = len(indices[indptr[target]:indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = indices[indptr[head]:indptr[head + 1]]
                nbrs_ce = get_cross_entropy_target_nbrs(target, nbrs, logits)
                if len(nbrs) == 0:
                    break
                one_ce_idx = nbrs_ce >= ce_limit
                if one_ce_idx.sum() >= topk:
                    nbrs = nbrs[one_ce_idx]
                    nbrs_ce = nbrs_ce[one_ce_idx]

                if is_topk and topk < len(nbrs):
                    idx_topk = np.argsort(nbrs_ce)[-topk:]
                    nbrs = nbrs[idx_topk]
                    nbrs_ce = nbrs_ce[idx_topk]

                idx = stochastic_accept(nbrs_ce)
                u = nbrs[idx]
                tmp_nodes.append(u)
                ce_limit = max(ce_limit, nbrs_ce[idx])
                if (u, head) not in tmp_edges:
                    tmp_edges[(head, u)] = 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    # 轮盘赌
    # 轮盘赌+topk
    @staticmethod
    @jit(cache=True, nopython=True, locals={'head': int64, 'u': int64})
    def deepwalk_purity_sample(purity, purity_r, labels, wrong_labels, indices, indptr, targets, sample_nums, is_topk):
        edges = []
        nodes = []

        for i, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            topk = len(indices[indptr[target]:indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = indices[indptr[head]:indptr[head + 1]]

                if len(nbrs) == 0:
                    break
                nbrs_purity = purity_r[nbrs]
                nbrs_labels = labels[nbrs]
                wrong_label_idx = nbrs_labels == wrong_labels[i]

                if wrong_label_idx.sum() >= topk:
                    nbrs = nbrs[wrong_label_idx]
                    nbrs_purity = purity[nbrs]

                if is_topk and topk < len(nbrs):
                    idx_topk = np.argsort(nbrs_purity)[-topk:]
                    nbrs = nbrs[idx_topk]
                    nbrs_purity = nbrs_purity[idx_topk]

                u = nbrs[stochastic_accept(nbrs_purity)]
                tmp_nodes.append(u)
                if (u, head) not in tmp_edges:
                    tmp_edges[(head, u)] = 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    @staticmethod
    @jit(cache=True, nopython=True, locals={'current_node': int64, 'u': int64})
    def deepwalk_biased_purity_sample(purity, purity_r, labels, wrong_labels, p, q, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        N = len(indptr) - 1
        for i, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            topk = len(indices[indptr[target]:indptr[target + 1]])
            current_node = target
            previous_node = N
            previous_node_neighbors = np.empty(0, dtype=np.int32)
            while len(tmp_nodes) < sample_nums:
                neighbors = indices[indptr[current_node]:indptr[current_node + 1]]
                if neighbors.size == 0:
                    break

                nbrs_purity = purity_r[neighbors]
                nbrs_labels = labels[neighbors]
                wrong_label_idx = nbrs_labels == wrong_labels[i]
                if wrong_label_idx.sum() >= topk:
                    neighbors = neighbors[wrong_label_idx]
                    nbrs_purity = purity[neighbors]

                probability = np.array([1 / q] * neighbors.size)
                probability[previous_node == neighbors] = 1 / p
                for i, nbr in enumerate(neighbors):
                    if np.any(nbr == previous_node_neighbors):
                        probability[i] = 1.

                probability = probability * nbrs_purity
                norm_probability = probability / np.sum(probability)

                u = random_choice(neighbors, norm_probability)
                tmp_nodes.append(u)
                if (u, current_node) not in tmp_edges:
                    tmp_edges[(current_node, u)] = 1
                previous_node_neighbors = neighbors
                previous_node = u
                current_node = u
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    # dw + 纯度幅度偏好
    def deepwalk_purity_gains_select_sample(self, targets, sample_nums, is_topk):
        edges = []
        nodes = []

        for target in targets:
            tmp_edges = {}
            tmp_nodes = [target]
            topk = len(self.indices[self.indptr[target]:self.indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]
                if len(nbrs) == 0:
                    break
                nbrs_purity_gains = get_purity_gains(self.indices, self.indptr, self.labels, self.purity,
                                                     target,
                                                     nbrs)
                nbrs_purity_gains = self.min_max_scaler.fit_transform(nbrs_purity_gains).ravel()
                nbrs_purity_gains = 1 - nbrs_purity_gains + self.eps
                if is_topk and topk < len(nbrs):
                    idx_topk = np.argsort(nbrs_purity_gains)[-topk:]
                    nbrs = nbrs[idx_topk]
                    nbrs_purity_gains = nbrs_purity_gains[idx_topk]

                u = nbrs[stochastic_accept(nbrs_purity_gains)]
                tmp_nodes.append(u)
                if (u, head) not in tmp_edges:
                    tmp_edges[(head, u)] = 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    @staticmethod
    @jit(cache=True, nopython=True, locals={'head': int32, 'u': int32})
    def deepwalk_purity_gains_sample(purity, labels, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        left = 1
        for target in targets:
            tmp_edges = {}
            tmp_nodes = [target]
            while len(tmp_nodes) < sample_nums:
                head = np.random.choice(np.array(tmp_nodes[-left:]))
                nbrs = indices[indptr[head]:indptr[head + 1]]

                if len(nbrs) == 0:
                    break
                nbrs_purity_gains = get_purity_gains(indices, indptr, labels, purity,
                                                     target,
                                                     nbrs)
                nbrs_purity_gains = nbrs_purity_gains.ravel()
                idx = nbrs_purity_gains <= 0
                if idx.sum() > 0:
                    nbrs = nbrs[idx]
                    left = len(nbrs)
                    for u in nbrs:
                        tmp_nodes.append(u)
                        if (u, head) not in tmp_edges:
                            tmp_edges[(head, u)] = 1
                else:
                    nbrs_purity_gains = nbrs_purity_gains[~idx]
                    nbrs = nbrs[~idx]
                    u = nbrs[nbrs_purity_gains.argmin()]
                    # nbrs_purity_gains = self.min_max_scaler.fit_transform(nbrs_purity_gains).ravel()
                    # nbrs_purity_gains = 1 - nbrs_purity_gains + self.eps
                    # u = nbrs[stochastic_accept(nbrs_purity_gains)]
                    left = 1
                    tmp_nodes.append(u)
                    if (u, head) not in tmp_edges:
                        tmp_edges[(head, u)] = 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    # 仅作保留
    def deepwalk_wl_kh_sample(self, targets, sample_nums):
        edges = []
        nodes = []
        for i, target in enumerate(targets):
            self.wl_limit = 0.5
            tmp_nodes, tmp_edges = get_hop_neighbors(self.indices, self.indptr, target, hops=2)
            nodes = list(nodes)
            while len(nodes) < sample_nums:
                head = np.random.choice(np.array(tmp_nodes))
                nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]
                if len(nbrs) == 0:
                    break
                nbrs_wl = self.wls[i][nbrs]
                one_wl_idx = nbrs_wl >= self.wl_limit
                if np.any(one_wl_idx):
                    nbrs = nbrs[one_wl_idx]
                    nbrs_wl = self.wl[nbrs]

                u = nbrs[stochastic_accept(nbrs_wl)]
                nodes.append(u)
                if (u, head) not in tmp_edges:
                    tmp_edges[(head, u)] = 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    def node2vec_sample(self, targets, sample_nums):
        '''
        Repeatedly simulate random walks from each node.
        '''

        edges = []
        nodes = []
        for i, target in enumerate(targets):
            if 'wl' in self.subgraph_type:
                alias_nodes = self.alias_nodes_[i]
                alias_edges = self.alias_edges_[i]
            else:
                alias_nodes = self.alias_nodes
                alias_edges = self.alias_edges
            tmp_edges = {}
            tmp_nodes = [target]
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]
                if len(nbrs) == 0:
                    break
                if len(tmp_nodes) == 1:
                    u = nbrs[alias_draw(alias_nodes[head][0], alias_nodes[head][1])]
                    tmp_nodes.append(u)
                    if (u, head) not in tmp_edges:
                        tmp_edges[(head, u)] = 1
                else:
                    prev = tmp_nodes[-2]
                    pos = (prev, head)
                    next = nbrs[alias_draw(alias_edges[pos][0], alias_edges[pos][1])]
                    tmp_nodes.append(next)
                    if (next, head) not in tmp_edges:
                        tmp_edges[(head, next)] = 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    def get_weight(self, dst, dst_nbr_idx):
        if self.subgraph_type == "n2v_wl":
            return self.wl_list[dst][dst_nbr_idx]
        elif self.subgraph_type == "n2v_purity":
            return self.purity_list[dst][dst_nbr_idx]
        elif self.subgraph_type == "n2v_ce":
            return self.ce_list[dst][dst_nbr_idx]

        return 1.0

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        p = self.p
        q = self.q

        unnormalized_probs = []
        nbrs = self.indices[self.indptr[dst]:self.indptr[dst + 1]]

        for dst_nbr_idx, dst_nbr in enumerate(nbrs):
            # p控制重复访问过的节点概率，若p校高，则访问刚刚访问过的节点src概率会变低
            if dst_nbr == src:
                unnormalized_probs.append(self.get_weight(dst, dst_nbr_idx) / p)
            elif self.adj_matrix[dst_nbr, src] != 0 or self.adj_matrix[src, dst_nbr] != 0:
                unnormalized_probs.append(self.get_weight(dst, dst_nbr_idx))
            else:
                # q控制BFS和DFS，若q>1，则倾向于访问和target接近的点（BFS），反之DFS
                unnormalized_probs.append(self.get_weight(dst, dst_nbr_idx) / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = np.array([float(u_prob) / norm_const for u_prob in unnormalized_probs])

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        N = self.adj_matrix.shape[0]

        alias_nodes = {}
        for node in range(N):
            nbrs = self.indices[self.indptr[node]:self.indptr[node + 1]]
            if self.subgraph_type == "n2v_purity":
                unnormalized_probs = self.purity_list[node]
            elif self.subgraph_type == "n2v_wl":
                unnormalized_probs = self.wl_list[node]
            elif self.subgraph_type == "n2v_ce":
                unnormalized_probs = self.ce_list[node]
            else:
                unnormalized_probs = [1 for _ in nbrs]

            norm_const = sum(unnormalized_probs)
            normalized_probs = np.array([float(u_prob) / norm_const for u_prob in unnormalized_probs])
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        for node in range(N):
            nbrs = self.indices[self.indptr[node]:self.indptr[node + 1]]
            for u in nbrs:
                alias_edges[(node, u)] = self.get_alias_edge(node, u)
                alias_edges[(u, node)] = self.get_alias_edge(u, node)

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


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
