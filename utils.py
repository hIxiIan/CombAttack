import torch
import numpy as np
import random
from graphgallery import functional as gf

def normalize_GCN(indices, weights, degree):
    row, col = indices
    inv_degree = torch.pow(degree, -0.5)
    normed_weights = weights * inv_degree[row] * inv_degree[col]
    return normed_weights


def ego_subgraph_spread_random(adj_matrix, targets, p, hops=1, hop_mode=False):
    if np.ndim(targets) == 0:
        targets = [targets]
    elif isinstance(targets, np.ndarray):
        targets = targets.tolist()
    else:
        targets = list(targets)

    indices = adj_matrix.indices  # 不为0的索引下标
    indptr = adj_matrix.indptr  # indices[indptr[i]:indptr[i+1]] 代表节点i的邻居索引下标

    edges = {}
    start = 0
    N = adj_matrix.shape[0]
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
                    if rd > p:
                        targets.append(u)
                if rd > p and (u, head) not in edges:
                    edges[(head, u)] = level + 1
            start += 1
        # 没有新节点加入
        if end == len(targets):
            break
        level = level + 1
        if hop_mode and level == hops:
            break
    e = []
    if hop_mode and len(targets[start:]):
        e = gf.extra_edges(indices, indptr, np.array(targets[start:]), seen, hops)
    if len(edges) == 0:
        assert False, 'edges are nil'
    #     print(edges)
    #     print(list(edges.keys()))
    #     print(list(edges.keys()) + e)
    return gf.asedge(list(edges.keys()) + e, shape='row_wise'), np.asarray(targets)