import random
import numpy as np
from graphgallery import functional as gf
from utils import get_purity, stochastic_accept, get_purity_martix, get_wl, get_wl_matrix, get_hop_neighbors, get_cross_entropy_matrix, get_target_subgraph_level


class Walker:
    def __init__(self, adj_matrix, labels, p=1.0, q=1.0, logits=None, is_purity_matrix=False, is_wl_matrix=False, is_ce_matrix=False, wl_limit=1.0, eps=1e-4):
        self.p = p
        self.q = q
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.adj_matrix = adj_matrix.tolil()  # weight默认为0/1
        self.adj_matrix_csr = adj_matrix
        self.labels = labels
        self.is_purity_matrix = is_purity_matrix
        self.is_wl_matrix = is_wl_matrix
        self.is_ce_matrix = is_ce_matrix
        self.purity = get_purity(adj_matrix.indices, adj_matrix.indptr, labels)  # 纯度
        self.purity_r = 1 - self.purity + eps  # 杂度
        self.purity += eps

        self.purity_matrix = get_purity_martix(self.purity, self.purity, labels) + eps
        self.purity_r_matrix = get_purity_martix(self.purity_r, self.purity_r, labels) + eps

        if is_purity_matrix and (self.p != 1.0 or self.q != 1.0):
            self.preprocess_transition_probs()

        if logits is not None:
            self.ce_matrix = get_cross_entropy_matrix(logits)
            if self.is_ce_matrix and (self.p != 1.0 or self.q != 1.0):
                self.preprocess_transition_probs()

        self.wrong_label = None
        self.wl = None
        self.wl_cnt = None
        self.wl_matrix = None
        self.wl_limit = wl_limit
        self.eps = eps

    def set_wrong_label(self, wrong_label):
        self.wrong_label = wrong_label
        self.wl, self.wl_cnt = get_wl(self.adj_matrix_csr.indices, self.adj_matrix_csr.indptr, self.labels, wrong_label, self.eps)
        self.wl_matrix = get_wl_matrix(self.wl)
        if self.is_wl_matrix and (self.p != 1.0 or self.q != 1.0):
            self.preprocess_transition_probs()

    # 纯随机游走
    def deepwalk_sample(self, targets, sample_nums):
        edges = {}
        nodes = []

        for target in targets:
            tmp_nodes = [target]
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]

                if len(nbrs) > 0:
                    u = random.choice(nbrs)
                    tmp_nodes.append(u)
                    if (u, head) not in edges:
                        edges[(head, u)] = 1
                else:
                    break
            nodes.extend(tmp_nodes)
        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(nodes)

    # 轮盘赌
    # 轮盘赌+topk
    # todo 尝试将度小考虑进来
    def deepwalk_wl_sample(self, targets, sample_nums, is_topk):
        edges = {}
        nodes = []

        for target in targets:
            tmp_nodes = [target]
            topk = len(self.indices[self.indptr[target]:self.indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]

                if len(nbrs) > 0:
                    nbrs_wl = self.wl[nbrs]
                    if is_topk and topk < len(nbrs):
                        idx_topk = np.argsort(nbrs_wl)[-topk:]
                        nbrs = nbrs[idx_topk]
                        nbrs_wl = nbrs_wl[idx_topk]

                    u = nbrs[stochastic_accept(nbrs_wl)]
                    tmp_nodes.append(u)
                    if (u, head) not in edges:
                        edges[(head, u)] = 1
                else:
                    break
            nodes.extend(tmp_nodes)
        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(nodes)

    # 轮盘赌+动态wrong_label阈值
    # 轮盘赌+动态wrong_label阈值+topk
    def deepwalk_wl_dynamic_sample(self, targets, sample_nums, is_topk):
        edges = {}
        nodes = []

        for target in targets:
            tmp_nodes = [target]
            wl_limit = 0.5
            topk = len(self.indices[self.indptr[target]:self.indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]
                if len(nbrs) > 0:
                    nbrs_wl = self.wl[nbrs]
                    one_wl_idx = nbrs_wl >= wl_limit
                    if any(one_wl_idx) and one_wl_idx.sum() >= topk:
                        nbrs = nbrs[one_wl_idx]
                        nbrs_wl = self.wl[nbrs]

                    if is_topk and topk < len(nbrs):
                        idx_topk = np.argsort(nbrs_wl)[-topk:]
                        nbrs = nbrs[idx_topk]
                        nbrs_wl = nbrs_wl[idx_topk]

                    u = nbrs[stochastic_accept(nbrs_wl)]
                    tmp_nodes.append(u)
                    wl_limit = max(wl_limit, self.wl[u])
                    if (u, head) not in edges:
                        edges[(head, u)] = 1
                else:
                    break
            nodes.extend(tmp_nodes)
        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(nodes)

    # 轮盘赌
    # 轮盘赌+topk
    def deepwalk_ce_sample(self, targets, sample_nums, is_topk):
        edges = {}
        nodes = []

        for target in targets:
            tmp_nodes = [target]
            topk = len(self.indices[self.indptr[target]:self.indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]
                if len(nbrs) > 0:
                    nbrs_ce = self.ce_matrix[target, nbrs]
                    if is_topk and topk < len(nbrs):
                        idx_topk = np.argsort(nbrs_ce)[-topk:]
                        nbrs = nbrs[idx_topk]
                        nbrs_ce = nbrs_ce[idx_topk]

                    u = nbrs[stochastic_accept(nbrs_ce)]
                    tmp_nodes.append(u)
                    if (u, head) not in edges:
                        edges[(head, u)] = 1
                else:
                    break
            nodes.extend(tmp_nodes)
        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(nodes)

    # 轮盘赌+动态cross_entropy阈值
    # 轮盘赌+动态cross_entropy阈值+topk
    def deepwalk_ce_dynamic_sample(self, targets, sample_nums, is_topk):
        edges = {}
        nodes = []

        for target in targets:
            tmp_nodes = [target]
            ce_limit = 1.0
            topk = len(self.indices[self.indptr[target]:self.indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]
                if len(nbrs) > 0:
                    nbrs_ce = self.ce_matrix[target, nbrs]
                    one_ce_idx = nbrs_ce >= ce_limit
                    if any(one_ce_idx) and one_ce_idx.sum() >= topk:
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
                    if (u, head) not in edges:
                        edges[(head, u)] = 1
                else:
                    break
            nodes.extend(tmp_nodes)
        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(nodes)

    # 轮盘赌
    # 轮盘赌+topk
    def deepwalk_purity_sample(self, targets, sample_nums, is_topk):
        edges = {}
        nodes = []

        for target in targets:
            tmp_nodes = [target]
            topk = len(self.indices[self.indptr[target]:self.indptr[target + 1]])
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]

                if len(nbrs) > 0:
                    nbrs_purity = self.purity_r[nbrs]
                    nbrs_labels = self.labels[nbrs]
                    wrong_label_idx = nbrs_labels == self.wrong_label

                    if any(wrong_label_idx) and wrong_label_idx.sum() >= topk:
                        nbrs = nbrs[wrong_label_idx]
                        nbrs_purity = self.purity[nbrs]

                    if is_topk and topk < len(nbrs):
                        idx_topk = np.argsort(nbrs_purity)[-topk:]
                        nbrs = nbrs[idx_topk]
                        nbrs_purity = nbrs_purity[idx_topk]

                    u = nbrs[stochastic_accept(nbrs_purity)]
                    tmp_nodes.append(u)
                    if (u, head) not in edges:
                        edges[(head, u)] = 1
                else:
                    break
            nodes.extend(tmp_nodes)
        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(nodes)

    # 仅作保留
    def deepwalk_sample_wl_keep_hops(self, targets, sample_nums):
        nodes, edges = get_hop_neighbors(self.adj_matrix_csr.indices, self.adj_matrix_csr.indptr, targets[0],
                                         hops=2)
        nodes = list(nodes)
        while len(nodes) < sample_nums:
            head = random.choice(nodes)
            nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]
            if len(nbrs) > 0:
                nbrs_wl = self.wl[nbrs]
                one_wl_idx = nbrs_wl >= self.wl_limit
                if any(one_wl_idx):
                    nbrs = nbrs[one_wl_idx]
                    nbrs_wl = self.wl[nbrs]

                u = nbrs[stochastic_accept(nbrs_wl)]
                nodes.append(u)
                if (u, head) not in edges:
                    edges[(head, u)] = 1
            else:
                break

        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(nodes)

    def node2vec_sample(self, targets, sample_nums):
        '''
        Repeatedly simulate random walks from each node.
        '''
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        edges = {}
        nodes = []
        for target in targets:
            tmp_nodes = [target]
            while len(tmp_nodes) < sample_nums:
                head = tmp_nodes[-1]
                nbrs = self.indices[self.indptr[head]:self.indptr[head + 1]]
                if len(nbrs) > 0:
                    if len(tmp_nodes) == 1:
                        u = nbrs[alias_draw(alias_nodes[head][0], alias_nodes[head][1])]
                        tmp_nodes.append(u)
                        if (u, head) not in edges:
                            edges[(head, u)] = 1
                    else:
                        prev = tmp_nodes[-2]
                        pos = (prev, head)
                        next = nbrs[alias_draw(alias_edges[pos][0], alias_edges[pos][1])]
                        tmp_nodes.append(next)
                        if (next, head) not in edges:
                            edges[(head, next)] = 1
                else:
                    break
            nodes.extend(tmp_nodes)
        return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(nodes)

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        p = self.p
        q = self.q

        unnormalized_probs = []
        nbrs = self.indices[self.indptr[dst]:self.indptr[dst + 1]]
        for dst_nbr in nbrs:
            if self.is_purity_matrix:
                # p控制重复访问过的节点概率，若p校高，则访问刚刚访问过的节点src概率会变低
                if dst_nbr == src:
                    unnormalized_probs.append(self.purity_r_matrix[dst][dst_nbr] / p)
                elif self.adj_matrix[dst_nbr, src] != 0 or self.adj_matrix[src, dst_nbr] != 0:
                    unnormalized_probs.append(self.purity_r_matrix[dst][dst_nbr])
                else:
                    # q控制BFS和DFS，若q>1，则倾向于访问和target接近的点（BFS），反之DFS
                    unnormalized_probs.append(self.purity_r_matrix[dst][dst_nbr] / q)
            elif self.is_wl_matrix:
                if dst_nbr == src:
                    unnormalized_probs.append(self.wl_matrix[dst][dst_nbr] / p)
                elif self.adj_matrix[dst_nbr, src] != 0 or self.adj_matrix[src, dst_nbr] != 0:
                    unnormalized_probs.append(self.wl_matrix[dst][dst_nbr])
                else:
                    unnormalized_probs.append(self.wl_matrix[dst][dst_nbr] / q)
            elif self.is_ce_matrix:
                if dst_nbr == src:
                    unnormalized_probs.append(self.ce_matrix[dst][dst_nbr] / p)
                elif self.adj_matrix[dst_nbr, src] != 0 or self.adj_matrix[src, dst_nbr] != 0:
                    unnormalized_probs.append(self.ce_matrix[dst][dst_nbr])
                else:
                    unnormalized_probs.append(self.ce_matrix[dst][dst_nbr] / q)
            else:
                if dst_nbr == src:
                    unnormalized_probs.append(1 / p)
                elif self.adj_matrix[dst_nbr, src] != 0 or self.adj_matrix[src, dst_nbr] != 0:
                    unnormalized_probs.append(1)
                else:
                    unnormalized_probs.append(1 / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        N = self.adj_matrix.shape[0]

        alias_nodes = {}
        for node in range(N):
            if self.is_purity_matrix:
                unnormalized_probs = [self.purity_r_matrix[node][nbr] for nbr in
                                      self.indices[self.indptr[node]:self.indptr[node + 1]]]
            elif self.is_wl_matrix:
                unnormalized_probs = [self.wl_matrix[node][nbr] for nbr in
                                      self.indices[self.indptr[node]:self.indptr[node + 1]]]
            elif self.is_ce_matrix:
                unnormalized_probs = [self.ce_matrix[node][nbr] for nbr in
                                      self.indices[self.indptr[node]:self.indptr[node + 1]]]
            else:
                unnormalized_probs = [1 for _ in self.indices[self.indptr[node]:self.indptr[node + 1]]]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        for node in range(N):
            nbr = self.indices[self.indptr[node]:self.indptr[node + 1]]
            for u in nbr:
                alias_edges[(node, u)] = self.get_alias_edge(node, u)
                alias_edges[(u, node)] = self.get_alias_edge(u, node)

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


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
