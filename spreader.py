import random
import numpy as np
from graphgallery import functional as gf
from utils import get_wl, get_wl_matrix


class Spreader:
    def __init__(self, adj_matrix, labels, wrong_label, prob, wl_limit=0.8, eps=1e-4):
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.adj_matrix = adj_matrix
        self.labels = labels
        self.wrong_label = wrong_label

        self.prob = prob
        self.wl_limit = wl_limit
        self.wl = get_wl(adj_matrix, labels, wrong_label) + eps
        self.wl_matrix = get_wl_matrix(self.wl)

    # 10^-3
    # gf.ego_graph是采样hops+1阶子图
    def spread_sample(self, targets, hops, keep_hops, sample_nums):
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

                for u in nbrs:
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
                            keep = False
                            if seen[u] < 0:
                                seen[u] = level + 1
                                if rd < self.prob:
                                    targets.append(u)
                                keep = u == nbrs[-1] and end == len(targets) and sample_nums > len(targets)
                                if keep:
                                    # 从根节点出发采样hops阶邻居个数为0，则在当前level随机选一个节点加入到候选集
                                    targets.append(random.choice(nbrs))
                            if ((rd < self.prob) or keep) and (u, head) not in edges:
                                edges[(head, u)] = level + 1
                start += 1

            level = level + 1
            # 达到所需节点数量 或者 没有新节点加入
            if sample_nums <= len(targets) or end == len(targets):
                break

        if len(edges) == 0:
            assert False, 'sample 0 edge'
        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets)
