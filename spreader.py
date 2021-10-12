import random
import numpy as np
from graphgallery import functional as gf
from utils import get_wl, get_wl_matrix, get_cross_entropy_matrix


class Spreader:
    def __init__(self, adj_matrix, labels, wrong_label, prob, logits=None, wl_limit=0.8, eps=1e-4):
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.adj_matrix = adj_matrix
        self.labels = labels
        self.wrong_label = wrong_label

        self.prob = prob
        self.wl_limit = wl_limit
        self.wl, self.wl_cnt = get_wl(adj_matrix, labels, wrong_label, eps)
        self.wl_matrix = get_wl_matrix(self.wl)

        self.ce_matrix = get_cross_entropy_matrix(logits)

    # 10^-3
    # todo:可能扩散不出去，措施直接与wrong_label相连？
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
                                    print(uu)
                                    if uu in targets and len(nbrs) > 1:
                                        uu = random.choice(nbrs[nbrs != uu])
                                        print(uu)
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
