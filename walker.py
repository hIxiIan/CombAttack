import random
import numpy as np
from graphgallery import functional as gf


class Walker:
    def __init__(self, adj_matrix, p=1.0, q=1.0):
        self.p = p
        self.q = q
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.adj_matrix = adj_matrix.tolil() # weight默认为0/1

    # 10^-3（p,q)初始化不影响deepwalk
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

    # 10^-3
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
            if dst_nbr == src:
                unnormalized_probs.append(1/p)
            elif self.adj_matrix[dst_nbr, src] != 0 or self.adj_matrix[src, dst_nbr] != 0:
                unnormalized_probs.append(1)
            else:
                unnormalized_probs.append(1/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        N = self.adj_matrix.shape[0]

        alias_nodes = {}
        for node in range(N):
            unnormalized_probs = [1 for _ in self.indices[self.indptr[node]:self.indptr[node + 1]]]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
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
        q[kk] = K*prob
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

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
