import random
import numpy as np
from graphgallery import functional as gf
from utils import get_wl, get_purity, get_cross_entropy_target_nbrs, get_purity_gains


class Spreader:
    def __init__(self, subgraph_type, adj_matrix, labels, prob, hops, logits=None, wl_limit=0.8, eps=1e-4):
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.adj_matrix = adj_matrix
        self.labels = labels
        self.subgraph_type = subgraph_type
        self.logits = logits

        self.hops = hops
        self.keep_hops = False
        self.prob = prob
        self.wrong_label = None
        self.wl = None
        self.wl_cnt = None
        self.wl_matrix = None
        self.wl_limit = wl_limit
        self.ce_list = None
        self.purity = None
        self.purity_r = None
        self.eps = eps

        self.init()

    def init(self):
        if "kh" in self.subgraph_type:
            self.keep_hops = True
        if "purity" in self.subgraph_type:
            self.purity = get_purity(self.indices, self.indptr, self.labels)  # 纯度
            self.purity_r = 1 - self.purity + self.eps  # 杂度
            self.purity += self.eps

    def set_wrong_label(self, wrong_label):
        self.wrong_label = wrong_label

        if "wl" in self.subgraph_type:
            self.wl, self.wl_cnt = get_wl(self.adj_matrix.indices, self.adj_matrix.indptr, self.labels, wrong_label, self.eps)

    # 10^-3
    def spread_random_wl_sample(self, targets, sample_nums):
        hops = self.hops
        keep_hops = self.keep_hops
        indices = self.indices
        indptr = self.indptr

        edges = {}
        start = 0
        N = self.adj_matrix.shape[0]
        # N个节点，全部初始化为-1
        seen = np.zeros(N) - 1
        seen[targets] = 0
        level = 0

        while True:
            end = len(targets)
            while start < end:
                head = targets[start]
                nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标

                for i, u in enumerate(nbrs):
                    # print((head, u))
                    if sample_nums <= len(targets):
                        break
                    if keep_hops and level < hops:
                        if seen[u] < 0:
                            seen[u] = level + 1
                            targets.append(u)
                        if (u, head) not in edges:
                            edges[(head, u)] = level + 1
                    else:
                        if self.labels[u] == self.wrong_label or self.wl[u] > self.wl_limit:
                            if seen[u] < 0:
                                seen[u] = level + 1
                                targets.append(u)
                            if (u, head) not in edges:
                                edges[(head, u)] = level + 1
                        else:
                            rd = random.random()
                            uu = u
                            if seen[uu] < 0:
                                seen[uu] = level + 1
                                if rd < self.prob:
                                    targets.append(uu)
                                if i == len(nbrs) - 1 and end == len(targets) and sample_nums > len(targets):
                                    # 从根节点出发采样hops阶邻居个数为0，则在当前level随机选一个节点加入到候选集
                                    while uu == head and len(nbrs) > 1:
                                        uu = random.choice(nbrs)
                                    targets.append(uu)
                            elif seen[uu] >= 0 and rd >= self.prob: # todo 有问题
                                # print('---')
                                if i == len(nbrs) - 1 and end == len(targets) and sample_nums > len(targets):
                                    # 从根节点出发采样hops阶邻居个数为0，则在当前level随机选一个节点加入到候选集
                                    # print(uu)
                                    if uu in targets and len(nbrs) > 1:
                                        uu = random.choice(nbrs[nbrs != uu])
                                        # print(uu)
                                    targets.append(uu)
                            if ((rd < self.prob) or (i == len(nbrs) - 1 and end == len(targets) and sample_nums > len(targets))) and (uu, head) not in edges:
                                edges[(head, uu)] = level + 1


                start += 1

                if sample_nums <= len(targets):
                    break

            level = level + 1
            # 达到所需节点数量 或者 没有新节点加入
            if sample_nums <= len(targets) or end == len(targets):
                break

        if len(edges) == 0:
            print('sample 0 edges')
            assert False, 'sample 0 edge'
        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets)

    # todo 可能有问题
    def spread_random_ce_sample(self, targets, sample_nums):
        hops = self.hops
        keep_hops = self.keep_hops
        indices = self.indices
        indptr = self.indptr

        edges = {}
        start = 0
        N = self.adj_matrix.shape[0]
        # N个节点，全部初始化为-1
        seen = np.zeros(N) - 1
        seen[targets] = 0
        level = 0
        root = targets[0]
        while True:
            end = len(targets)

            while start < end:
                head = targets[start]
                nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标

                for i, u in enumerate(nbrs):
                    if sample_nums <= len(targets):
                        break
                    if keep_hops and level < hops:
                        if seen[u] < 0:
                            seen[u] = level + 1
                            targets.append(u)
                        if (u, head) not in edges:
                            edges[(head, u)] = level + 1
                    else:
                        nbrs_ce = get_cross_entropy_target_nbrs(root, nbrs, self.logits)
                        ce_limit = nbrs_ce.mean()
                        if nbrs_ce[i] > ce_limit:
                            if seen[u] < 0:
                                seen[u] = level + 1
                                targets.append(u)
                            if (u, head) not in edges:
                                edges[(head, u)] = level + 1
                        else:
                            rd = random.random()
                            uu = u
                            # print((head, uu), rd, self.prob)
                            # print(seen[uu])
                            if seen[uu] < 0:
                                seen[uu] = level + 1
                                if rd < self.prob:
                                    targets.append(uu)
                                if i == len(nbrs) - 1 and end == len(targets) and sample_nums > len(targets):
                                    # 从根节点出发采样hops阶邻居个数为0，则在当前level随机选一个节点加入到候选集
                                    while uu == head and len(nbrs) > 1:
                                        uu = random.choice(nbrs)
                                    targets.append(uu)
                            elif seen[uu] >= 0 and rd >= self.prob: # todo 有问题
                                # print('---')
                                if i == len(nbrs) - 1 and end == len(targets) and sample_nums > len(targets):
                                    # 从根节点出发采样hops阶邻居个数为0，则在当前level随机选一个节点加入到候选集
                                    # print(uu)
                                    if uu in targets and len(nbrs) > 1:
                                        uu = random.choice(nbrs[nbrs != uu])
                                        # print(uu)
                                    targets.append(uu)
                            if ((rd < self.prob) or (i == len(nbrs) - 1 and end == len(targets) and sample_nums > len(targets))) and (uu, head) not in edges:
                                edges[(head, uu)] = level + 1


                start += 1

                if sample_nums <= len(targets):
                    break

            level = level + 1
            # 达到所需节点数量 或者 没有新节点加入
            if sample_nums <= len(targets) or end == len(targets):
                break

        if len(edges) == 0:
            print('sample 0 edges')
            assert False, 'sample 0 edge'
        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets)

    def spread_ce_sample(self, targets, sample_nums):
        hops = self.hops
        keep_hops = self.keep_hops
        indices = self.indices
        indptr = self.indptr

        edges = {}
        start = 0
        N = self.adj_matrix.shape[0]
        # N个节点，全部初始化为-1
        seen = np.zeros(N) - 1
        seen[targets] = 0
        level = 0
        root = targets[0]
        while sample_nums > len(targets):
            end = len(targets)
            if start == end:
                break
            target_topk = len(self.indices[self.indptr[root]:self.indptr[root + 1]])
            while start < end and sample_nums > len(targets):
                head = targets[start]
                nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标

                if keep_hops and level < hops:
                    for i, u in enumerate(nbrs):
                        if sample_nums <= len(targets):
                            break
                        if seen[u] < 0:
                            seen[u] = level + 1
                            targets.append(u)
                        if (u, head) not in edges:
                            edges[(head, u)] = level + 1
                    start += 1
                    continue
                nbrs_ce = get_cross_entropy_target_nbrs(root, nbrs, self.logits)
                ce_limit = np.percentile(nbrs_ce, 50)
                idx_ce = nbrs_ce > ce_limit
                if idx_ce.sum() >= target_topk:
                    nbrs = nbrs[idx_ce]
                    nbrs_ce = nbrs_ce[idx_ce]

                if sample_nums < len(targets) + len(nbrs):
                    topk = sample_nums - len(targets)
                    idx_topk = np.argsort(nbrs_ce)[-topk:]
                    nbrs = nbrs[idx_topk]

                for i, u in enumerate(nbrs):
                    if seen[u] < 0:
                        seen[u] = level + 1
                        targets.append(u)
                    if (u, head) not in edges:
                        edges[(head, u)] = level + 1
                start += 1

            level = level + 1

        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets)

    def spread_wl_sample(self, targets, sample_nums):
        hops = self.hops
        keep_hops = self.keep_hops
        indices = self.indices
        indptr = self.indptr

        edges = {}
        start = 0
        N = self.adj_matrix.shape[0]
        # N个节点，全部初始化为-1
        seen = np.zeros(N) - 1
        seen[targets] = 0
        level = 0
        root = targets[0]
        while sample_nums > len(targets):
            end = len(targets)
            if start == end:
                break
            wl_limit = 0.5
            target_topk = len(self.indices[self.indptr[root]:self.indptr[root + 1]])
            while start < end and sample_nums > len(targets):
                head = targets[start]
                nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标
                if keep_hops and level < hops:
                    for i, u in enumerate(nbrs):
                        if sample_nums <= len(targets):
                            break
                        if seen[u] < 0:
                            seen[u] = level + 1
                            targets.append(u)
                        if (u, head) not in edges:
                            edges[(head, u)] = level + 1
                    start += 1
                    continue
                nbrs_wl = self.wl[nbrs]
                idx_wl = nbrs_wl > wl_limit
                if idx_wl.sum() >= target_topk:
                    nbrs = nbrs[idx_wl]
                    nbrs_wl = nbrs_wl[idx_wl]

                if sample_nums < len(targets) + len(nbrs):
                    topk = sample_nums - len(targets)
                    idx_topk = np.argsort(nbrs_wl)[-topk:]
                    nbrs = nbrs[idx_topk]

                for i, u in enumerate(nbrs):
                    if seen[u] < 0:
                        seen[u] = level + 1
                        targets.append(u)
                        wl_limit = max(wl_limit, self.wl[u])
                    if (u, head) not in edges:
                        edges[(head, u)] = level + 1
                start += 1

            level = level + 1

        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets)

    def spread_wl_improve_sample(self, targets, sample_nums):
        hops = self.hops
        keep_hops = self.keep_hops
        indices = self.indices
        indptr = self.indptr

        edges = {}
        start = 0
        N = self.adj_matrix.shape[0]
        # N个节点，全部初始化为-1
        seen = np.zeros(N) - 1
        seen[targets] = 0
        level = 0
        root = targets[0]
        while sample_nums > len(targets):
            end = len(targets)
            if start == end:
                break
            wl_limit = 0.5
            target_topk = len(self.indices[self.indptr[root]:self.indptr[root + 1]])
            while start < end and sample_nums > len(targets):
                head = targets[start]
                nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标
                if len(nbrs) > 0:
                    if keep_hops and level < hops:
                        for i, u in enumerate(nbrs):
                            if sample_nums <= len(targets):
                                break
                            if seen[u] < 0:
                                seen[u] = level + 1
                                targets.append(u)
                            if (u, head) not in edges:
                                edges[(head, u)] = level + 1
                        start += 1
                        continue
                    nbrs_wl = self.wl[nbrs]
                    idx_wl = nbrs_wl > wl_limit
                    if idx_wl.sum() >= target_topk:
                        nbrs = nbrs[idx_wl]
                        nbrs_wl = nbrs_wl[idx_wl]

                        if sample_nums < len(targets) + len(nbrs):
                            topk = sample_nums - len(targets)
                            idx_topk = np.argsort(nbrs_wl)[-topk:]
                            nbrs = nbrs[idx_topk]
                            nbrs_wl = nbrs_wl[idx_topk]

                        next_start_offset = 1
                        ccc = 0
                        for i, u in enumerate(nbrs):
                            if seen[u] < 0:
                                ccc += 1
                                seen[u] = level + 1
                                targets.append(u)
                                if wl_limit > self.wl[u]:
                                    wl_limit = self.wl[u]
                                    next_start_offset = ccc
                            if (u, head) not in edges:
                                edges[(head, u)] = level + 1
                        start += next_start_offset
                    else:
                        if sample_nums < len(targets) + len(nbrs):
                            topk = sample_nums - len(targets)
                            idx_topk = np.argsort(nbrs_wl)[-topk:]
                            nbrs = nbrs[idx_topk]

                        next_start_offset = 1
                        ccc = 0
                        for i, u in enumerate(nbrs):
                            if random.random() < 0.5:
                                if seen[u] < 0:
                                    ccc += 1
                                    seen[u] = level + 1
                                    targets.append(u)
                                    if wl_limit > self.wl[u]:
                                        wl_limit = self.wl[u]
                                        next_start_offset = ccc
                                if (u, head) not in edges:
                                    edges[(head, u)] = level + 1
                        start += next_start_offset
                else:
                    start += 1
                    break

            level = level + 1

        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets)

    def spread_random_purity_gains_sample(self, targets, sample_nums):
        hops = self.hops
        keep_hops = self.keep_hops
        indices = self.indices
        indptr = self.indptr

        edges = {}
        start = 0
        N = self.adj_matrix.shape[0]
        # N个节点，全部初始化为-1
        seen = np.zeros(N) - 1
        seen[targets] = 0
        level = 0
        root = targets[0]
        while sample_nums > len(targets):
            end = len(targets)
            if start >= end:
                break
            target_topk = len(self.indices[self.indptr[root]:self.indptr[root + 1]])
            while start < end and sample_nums > len(targets):
                head = targets[start]
                nbrs = indices[indptr[head]:indptr[head + 1]]  # 节点head的邻居索引下标
                if len(nbrs) > 0:
                    if keep_hops and level < hops:
                        for i, u in enumerate(nbrs):
                            if sample_nums <= len(targets):
                                break
                            if seen[u] < 0:
                                seen[u] = level + 1
                                targets.append(u)
                            if (u, head) not in edges:
                                edges[(head, u)] = level + 1
                        start += 1
                        continue
                    nbrs_purity_gains = get_purity_gains(self.indices, self.indptr, self.labels, self.purity, root, nbrs)
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
                                targets.append(u)
                                if nbrs_purity_gains_ravel[i] < _min:
                                    _min = nbrs_purity_gains_ravel[i]
                                    next_start_offset = ccc
                            if (u, head) not in edges:
                                edges[(head, u)] = level + 1
                        start += next_start_offset
                    else:
                        for i, u in enumerate(nbrs):
                            if random.random() < self.prob:
                                if seen[u] < 0:
                                    seen[u] = level + 1
                                    targets.append(u)
                                if (u, head) not in edges:
                                    edges[(head, u)] = level + 1
                        # if sample_nums < len(targets) + len(nbrs):
                        #     topk = sample_nums - len(targets)
                        #     idx_topk = np.argsort(nbrs_purity_gains)[-topk:]
                        #     nbrs = nbrs[idx_topk]
                        start += 1
                else:
                    start += 1
                    break
            level = level + 1

        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets)