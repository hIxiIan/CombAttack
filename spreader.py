import random
import numpy as np
from graphgallery import functional as gf


class Spreader:
    def __init__(self, adj_matrix):
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.adj_matrix = adj_matrix

    # 10^-3
    def spread_sample(self, targets, prob, hops, hop_mode):
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
                    # rd有可能导致edges为空，需要考虑如何避免这个问题
                    # 与SGA比较的时候，需要固定随机种子比较
                    rd = random.random()
                    if seen[u] < 0:
                        seen[u] = level + 1
                        if rd < prob:
                            targets.append(u)
                        elif u == nbrs[-1] and level < hops and end == len(targets):
                            # 从根节点出发采样hops阶邻居个数为0，则在当前level随机选一个节点加入到候选集
                            targets.append(random.choice(nbrs))
                    if ((rd < prob) or (u == nbrs[-1] and level < hops and end == len(targets))) and (
                            u, head) not in edges:
                        edges[(head, u)] = level + 1
                start += 1

            level = level + 1
            # 没有新节点加入
            if (end == len(targets)) or (hop_mode and level == hops):
                break
        e = []
        if hop_mode and len(targets[start:]):
            e = gf.extra_edges(indices, indptr, np.array(targets[start:]), seen, hops)
        if len(edges) == 0:
            assert False, 'sample 0 edge'
        return gf.asedge(list(edges.keys()) + e, shape='row_wise'), np.asarray(targets)