import random
import numpy as np
from graphgallery import functional as gf
from utils import get_wl, get_purity, get_cross_entropy_target_nbrs, get_purity_gains
from numba import jit


class Spreader:
    def __init__(self, targets, sample_ratio, subgraph_type, adj_matrix, labels, prob, hops, ori_logits=None, logits=None, wl_limit=0.5, eps=1e-4):
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.adj_matrix = adj_matrix
        self.labels = labels
        self.subgraph_type = subgraph_type
        self.ori_logits = ori_logits
        self.logits = logits

        self.hops = hops
        self.keep_hops = False
        self.prob = prob
        self.wl = None
        self.wl_limit = wl_limit
        self.ce_list = None
        self.purity = None
        self.purity_r = None
        self.eps = eps

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
        self.wl_lists = None

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

        self.wls = np.array(wls)
        self.wl_cnts = np.array(wl_cnts)

    def get_all_purity(self):
        self.purity = get_purity(self.indices, self.indptr, self.labels)  # 纯度
        self.purity_r = 1 - self.purity + self.eps  # 杂度
        self.purity += self.eps

    def spread_walk(self):
        if "kh" in self.subgraph_type:
            self.keep_hops = True
        if 'wl' in self.subgraph_type:
            self.get_all_wl()
            if self.subgraph_type in ['spread_random_wl', 'spread_random_wl_kh']:
                sub_edges, sub_nodes = self.spread_random_wl_sample(self.wls, self.wl_limit, self.labels, self.wrong_labels, self.hops, self.keep_hops, self.prob, self.indices, self.indptr, self.targets, self.sample_nums)
            elif self.subgraph_type in ['spread_wl', 'spread_wl_kh']:
                sub_edges, sub_nodes = self.spread_wl_sample(self.wls, self.hops, self.keep_hops, self.indices, self.indptr, self.targets, self.sample_nums)
            elif self.subgraph_type == 'spread_wl_improve':
                sub_edges, sub_nodes = self.spread_wl_improve_sample(self.wls, self.hops, self.keep_hops, self.prob, self.indices, self.indptr, self.targets, self.sample_nums)
        elif 'ce' in self.subgraph_type:
            if self.subgraph_type in ['spread_random_ce', 'spread_random_ce_kh']:
                sub_edges, sub_nodes = self.spread_random_ce_sample(self.logits, self.hops, self.keep_hops, self.prob, self.indices, self.indptr, self.targets, self.sample_nums)
            elif self.subgraph_type in ['spread_ce', 'spread_ce_kh']:
                sub_edges, sub_nodes = self.spread_ce_sample(self.logits, self.hops, self.keep_hops, self.indices, self.indptr, self.targets, self.sample_nums)
        elif 'purity' in self.subgraph_type:
            self.get_all_purity()
            if self.subgraph_type == 'spread_random_purity_gains':
                sub_edges, sub_nodes = self.spread_random_purity_gains_sample(self.purity, self.labels, self.hops, self.keep_hops, self.prob, self.indices, self.indptr, self.targets, self.sample_nums)

        self.sample_edges = [gf.asedge(sub_edge, shape='row_wise') for sub_edge in sub_edges]
        self.sample_nodes = [np.unique(sub_node) for sub_node in sub_nodes]

    # 10^-3
    @staticmethod
    @jit(cache=True, nopython=True)
    def spread_random_wl_sample(wls, wl_limit, labels, wrong_labels, hops, keep_hops, prob, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        for ti, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            start = 0
            N = len(indptr) - 1
            # N个节点，全部初始化为-1
            seen = np.zeros(N) - 1
            seen[target] = 0
            level = 0
            while True:
                end = len(tmp_nodes)
                while start < end:
                    head = tmp_nodes[start]
                    nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标

                    for i, u in enumerate(nbrs):
                        if sample_nums <= len(tmp_nodes):
                            break
                        if keep_hops and level < hops:
                            if seen[u] < 0:
                                seen[u] = level + 1
                                tmp_nodes.append(u)
                            if (u, head) not in tmp_edges:
                                tmp_edges[(head, u)] = level + 1
                        else:
                            if labels[u] == wrong_labels[ti] or wls[ti][u] > wl_limit:
                                if seen[u] < 0:
                                    seen[u] = level + 1
                                    tmp_nodes.append(u)
                                if (u, head) not in tmp_edges:
                                    tmp_edges[(head, u)] = level + 1
                            else:
                                rd = np.random.random()
                                uu = u
                                if seen[uu] < 0:
                                    seen[uu] = level + 1
                                    if rd < prob:
                                        tmp_nodes.append(uu)
                                    if i == len(nbrs) - 1 and end == len(tmp_nodes) and sample_nums > len(tmp_nodes):
                                        # 从根节点出发采样hops阶邻居个数为0，则在当前level随机选一个节点加入到候选集
                                        while uu == head and len(nbrs) > 1:
                                            uu = np.random.choice(nbrs)
                                        tmp_nodes.append(uu)
                                elif seen[uu] >= 0 and rd >= prob: # todo 有问题
                                    if i == len(nbrs) - 1 and end == len(tmp_nodes) and sample_nums > len(tmp_nodes):
                                        # 从根节点出发采样hops阶邻居个数为0，则在当前level随机选一个节点加入到候选集
                                        if uu in tmp_nodes and len(nbrs) > 1:
                                            uu = np.random.choice(nbrs[nbrs != uu])
                                        tmp_nodes.append(uu)
                                if ((rd < prob) or (i == len(nbrs) - 1 and end == len(tmp_nodes) and sample_nums > len(tmp_nodes))) and (uu, head) not in tmp_edges:
                                    tmp_edges[(head, uu)] = level + 1

                    start += 1

                    if sample_nums <= len(tmp_nodes):
                        break

                level = level + 1
                # 达到所需节点数量 或者 没有新节点加入
                if sample_nums <= len(tmp_nodes) or end == len(tmp_nodes):
                    break
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    # todo 可能有问题
    @staticmethod
    @jit(cache=True, nopython=True)
    def spread_random_ce_sample(logits, hops, keep_hops, prob, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        for _, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            start = 0
            N = len(indptr) - 1
            seen = np.zeros(N) - 1
            seen[target] = 0
            level = 0
            while True:
                end = len(tmp_nodes)
                while start < end:
                    head = tmp_nodes[start]
                    nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标

                    for i, u in enumerate(nbrs):
                        if sample_nums <= len(tmp_nodes):
                            break
                        if keep_hops and level < hops:
                            if seen[u] < 0:
                                seen[u] = level + 1
                                tmp_nodes.append(u)
                            if (u, head) not in tmp_edges:
                                tmp_edges[(head, u)] = level + 1
                        else:
                            nbrs_ce = get_cross_entropy_target_nbrs(target, nbrs, logits)
                            ce_limit = nbrs_ce.mean()
                            if nbrs_ce[i] > ce_limit:
                                if seen[u] < 0:
                                    seen[u] = level + 1
                                    tmp_nodes.append(u)
                                if (u, head) not in tmp_edges:
                                    tmp_edges[(head, u)] = level + 1
                            else:
                                rd = np.random.random()
                                uu = u
                                if seen[uu] < 0:
                                    seen[uu] = level + 1
                                    if rd < prob:
                                        tmp_nodes.append(uu)
                                    if i == len(nbrs) - 1 and end == len(tmp_nodes) and sample_nums > len(tmp_nodes):
                                        # 从根节点出发采样hops阶邻居个数为0，则在当前level随机选一个节点加入到候选集
                                        while uu == head and len(nbrs) > 1:
                                            uu = np.random.choice(nbrs)
                                        tmp_nodes.append(uu)
                                elif seen[uu] >= 0 and rd >= prob: # todo 有问题
                                    if i == len(nbrs) - 1 and end == len(tmp_nodes) and sample_nums > len(tmp_nodes):
                                        # 从根节点出发采样hops阶邻居个数为0，则在当前level随机选一个节点加入到候选集
                                        if uu in tmp_nodes and len(nbrs) > 1:
                                            uu = np.random.choice(nbrs[nbrs != uu])
                                        tmp_nodes.append(uu)
                                if ((rd < prob) or (i == len(nbrs) - 1 and end == len(tmp_nodes) and sample_nums > len(tmp_nodes))) and (uu, head) not in tmp_edges:
                                    tmp_edges[(head, uu)] = level + 1

                    start += 1

                    if sample_nums <= len(tmp_nodes):
                        break

                level = level + 1
                # 达到所需节点数量 或者 没有新节点加入
                if sample_nums <= len(tmp_nodes) or end == len(tmp_nodes):
                    break
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    @staticmethod
    @jit(cache=True, nopython=True)
    def spread_ce_sample(logits, hops, keep_hops, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        for _, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            start = 0
            N = len(indptr) - 1
            # N个节点，全部初始化为-1
            seen = np.zeros(N) - 1
            seen[target] = 0
            level = 0
            while sample_nums > len(tmp_nodes):
                end = len(tmp_nodes)
                if start == end:
                    break
                target_topk = len(indices[indptr[target]:indptr[target + 1]])
                while start < end and sample_nums > len(tmp_nodes):
                    head = tmp_nodes[start]
                    nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标

                    if keep_hops and level < hops:
                        for i, u in enumerate(nbrs):
                            if sample_nums <= len(tmp_nodes):
                                break
                            if seen[u] < 0:
                                seen[u] = level + 1
                                tmp_nodes.append(u)
                            if (u, head) not in tmp_edges:
                                tmp_edges[(head, u)] = level + 1
                        start += 1
                        continue
                    nbrs_ce = get_cross_entropy_target_nbrs(target, nbrs, logits)
                    ce_limit = np.percentile(nbrs_ce, 50)
                    idx_ce = nbrs_ce > ce_limit
                    if idx_ce.sum() >= target_topk:
                        nbrs = nbrs[idx_ce]
                        nbrs_ce = nbrs_ce[idx_ce]

                    if sample_nums < len(tmp_nodes) + len(nbrs):
                        topk = sample_nums - len(tmp_nodes)
                        idx_topk = np.argsort(nbrs_ce)[-topk:]
                        nbrs = nbrs[idx_topk]

                    for i, u in enumerate(nbrs):
                        if seen[u] < 0:
                            seen[u] = level + 1
                            tmp_nodes.append(u)
                        if (u, head) not in tmp_edges:
                            tmp_edges[(head, u)] = level + 1
                    start += 1

                level = level + 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    @staticmethod
    @jit(cache=True, nopython=True)
    def spread_wl_sample(wls, hops, keep_hops, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        for ti, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            start = 0
            N = len(indptr) - 1
            # N个节点，全部初始化为-1
            seen = np.zeros(N) - 1
            seen[target] = 0
            level = 0
            while sample_nums > len(tmp_nodes):
                end = len(tmp_nodes)
                if start == end:
                    break
                wl_limit = 0.5
                target_topk = len(indices[indptr[target]:indptr[target + 1]])
                while start < end and sample_nums > len(tmp_nodes):
                    head = tmp_nodes[start]
                    nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标
                    if keep_hops and level < hops:
                        for i, u in enumerate(nbrs):
                            if sample_nums <= len(tmp_nodes):
                                break
                            if seen[u] < 0:
                                seen[u] = level + 1
                                tmp_nodes.append(u)
                            if (u, head) not in tmp_edges:
                                tmp_edges[(head, u)] = level + 1
                        start += 1
                        continue
                    nbrs_wl = wls[ti][nbrs]
                    idx_wl = nbrs_wl > wl_limit
                    if idx_wl.sum() >= target_topk:
                        nbrs = nbrs[idx_wl]
                        nbrs_wl = nbrs_wl[idx_wl]

                    if sample_nums < len(tmp_nodes) + len(nbrs):
                        topk = sample_nums - len(tmp_nodes)
                        idx_topk = np.argsort(nbrs_wl)[-topk:]
                        nbrs = nbrs[idx_topk]

                    for i, u in enumerate(nbrs):
                        if seen[u] < 0:
                            seen[u] = level + 1
                            tmp_nodes.append(u)
                            wl_limit = max(wl_limit, wls[ti][u])
                        if (u, head) not in tmp_edges:
                            tmp_edges[(head, u)] = level + 1
                    start += 1

                level = level + 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    @staticmethod
    @jit(cache=True, nopython=True)
    def spread_wl_improve_sample(wls, hops, keep_hops, prob, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        for ti, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            start = 0
            N = len(indptr) - 1
            seen = np.zeros(N) - 1
            seen[target] = 0
            level = 0
            while sample_nums > len(tmp_nodes):
                end = len(tmp_nodes)
                if start == end:
                    break
                wl_limit = 0.5
                target_topk = len(indices[indptr[target]:indptr[target + 1]])
                while start < end and sample_nums > len(tmp_nodes):
                    head = tmp_nodes[start]
                    nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标
                    if len(nbrs) > 0:
                        if keep_hops and level < hops:
                            for i, u in enumerate(nbrs):
                                if sample_nums <= len(tmp_nodes):
                                    break
                                if seen[u] < 0:
                                    seen[u] = level + 1
                                    tmp_nodes.append(u)
                                if (u, head) not in tmp_edges:
                                    tmp_edges[(head, u)] = level + 1
                            start += 1
                            continue
                        nbrs_wl = wls[ti][nbrs]
                        idx_wl = nbrs_wl > wl_limit
                        if idx_wl.sum() >= target_topk:
                            nbrs = nbrs[idx_wl]
                            nbrs_wl = nbrs_wl[idx_wl]

                            if sample_nums < len(tmp_nodes) + len(nbrs):
                                topk = sample_nums - len(tmp_nodes)
                                idx_topk = np.argsort(nbrs_wl)[-topk:]
                                nbrs = nbrs[idx_topk]
                                nbrs_wl = nbrs_wl[idx_topk]

                            next_start_offset = 1
                            ccc = 0
                            for i, u in enumerate(nbrs):
                                if seen[u] < 0:
                                    ccc += 1
                                    seen[u] = level + 1
                                    tmp_nodes.append(u)
                                    if wl_limit > wls[ti][u]:
                                        wl_limit = wls[ti][u]
                                        next_start_offset = ccc
                                if (u, head) not in tmp_edges:
                                    tmp_edges[(head, u)] = level + 1
                            start += next_start_offset
                        else:
                            if sample_nums < len(tmp_nodes) + len(nbrs):
                                topk = sample_nums - len(tmp_nodes)
                                idx_topk = np.argsort(nbrs_wl)[-topk:]
                                nbrs = nbrs[idx_topk]

                            next_start_offset = 1
                            ccc = 0
                            for i, u in enumerate(nbrs):
                                if random.random() < prob:
                                    if seen[u] < 0:
                                        ccc += 1
                                        seen[u] = level + 1
                                        tmp_nodes.append(u)
                                        if wl_limit > wls[ti][u]:
                                            wl_limit = wls[ti][u]
                                            next_start_offset = ccc
                                    if (u, head) not in tmp_edges:
                                        tmp_edges[(head, u)] = level + 1
                            start += next_start_offset
                    else:
                        start += 1
                        break

                level = level + 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes

    @staticmethod
    @jit(cache=True, nopython=True)
    def spread_random_purity_gains_sample(purity, labels, hops, keep_hops, prob, indices, indptr, targets, sample_nums):
        edges = []
        nodes = []
        for _, target in enumerate(targets):
            tmp_edges = {}
            tmp_nodes = [target]
            start = 0
            N = len(indptr) - 1
            # N个节点，全部初始化为-1
            seen = np.zeros(N) - 1
            seen[target] = 0
            level = 0
            while sample_nums > len(tmp_nodes):
                end = len(tmp_nodes)
                if start >= end:
                    break
                target_topk = len(indices[indptr[target]:indptr[target + 1]])
                while start < end and sample_nums > len(tmp_nodes):
                    head = tmp_nodes[start]
                    nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标
                    if len(nbrs) > 0:
                        if keep_hops and level < hops:
                            for i, u in enumerate(nbrs):
                                if sample_nums <= len(tmp_nodes):
                                    break
                                if seen[u] < 0:
                                    seen[u] = level + 1
                                    tmp_nodes.append(u)
                                if (u, head) not in tmp_edges:
                                    tmp_edges[(head, u)] = level + 1
                            start += 1
                            continue
                        nbrs_purity_gains = get_purity_gains(indices, indptr, labels, purity, target, nbrs)
                        nbrs_purity_gains_ravel = nbrs_purity_gains.ravel()
                        idx = nbrs_purity_gains_ravel <= 0
                        if idx.sum() > 0:
                            nbrs = nbrs[idx]
                            nbrs_purity_gains_ravel = nbrs_purity_gains_ravel[idx]
                            ccc = 0
                            next_start_offset = 1
                            _min = 1.1
                            for i, u in enumerate(nbrs):
                                if seen[u] < 0:
                                    ccc += 1
                                    seen[u] = level + 1
                                    tmp_nodes.append(u)
                                    if nbrs_purity_gains_ravel[i] < _min:
                                        _min = nbrs_purity_gains_ravel[i]
                                        next_start_offset = ccc
                                if (u, head) not in tmp_edges:
                                    tmp_edges[(head, u)] = level + 1
                            start += next_start_offset
                        else:
                            for i, u in enumerate(nbrs):
                                if random.random() < prob:
                                    if seen[u] < 0:
                                        seen[u] = level + 1
                                        tmp_nodes.append(u)
                                    if (u, head) not in tmp_edges:
                                        tmp_edges[(head, u)] = level + 1
                            # if sample_nums < len(targets) + len(nbrs):
                            #     topk = sample_nums - len(targets)
                            #     idx_topk = np.argsort(nbrs_purity_gains)[-topk:]
                            #     nbrs = nbrs[idx_topk]
                            start += 1
                    else:
                        start += 1
                        break
                level = level + 1
            edges.append(list(tmp_edges.keys()))
            nodes.append(tmp_nodes)
        return edges, nodes