import types

import numba
import numpy as np
from numba import int64

from graphgallery import functional as gf
from numba.typed import Dict
from numba.core import types


class PPRer:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.out_degree = np.sum(adj_matrix > 0, axis=1).A1

    # alpha >> 1 pay more attention to immediate neighbors
    # alpha >> 0 pay more attention to multi-hop neighbors
    def ppr_sample(self, targets, alpha, epsilon):
        edges, nodes, weights = calc_ppr(self.indptr, self.indices, self.out_degree, alpha, epsilon, np.asarray(targets))
        _edges = update_edges(edges)
        return gf.asedge(list(_edges.keys()), shape='row_wise'), nodes

    def ppr_topk_sample(self, targets, alpha, epsilon, topk):
        edges, nodes, weights, del_nodes = calc_ppr_topk(self.indptr, self.indices, self.out_degree, alpha, epsilon, np.asarray(targets), topk)
        delete_edges(edges, del_nodes) # 好像有问题
        _edges = update_edges(edges)
        return gf.asedge(list(_edges.keys()), shape='row_wise'), np.asarray(nodes).ravel()

    def ppr_topk_sample_parallel(self, targets, alpha, epsilon, topk):
        edges, nodes, weights, del_nodes = calc_ppr_topk_parallel(self.indptr, self.indices, self.out_degree, alpha, epsilon, np.asarray(targets), topk)
        delete_edges(edges, del_nodes)
        _edges = update_edges(edges)
        return gf.asedge(list(_edges.keys()), shape='row_wise'), np.asarray(nodes).ravel()


# @numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    edges = {}
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)
                    _vnode = int64(vnode)
                    _unode = int64(unode)
                    if (_vnode, _unode) not in edges:
                        edges[(_unode, _vnode)] = 1

    return list(p.keys()), list(p.values()), edges


def update_edges(edges):
    _edges = {}
    for edge in edges:
        _edges.update(edge)
    return _edges


def delete_edges(edges, del_nodes):
    for i in range(len(edges)):
        dict_del_key(edges[i], del_nodes[i])


def dict_del_key(_dict, _del_keys):
    keys = list(_dict.keys())
    for key in keys:
        if key[0] in _del_keys or key[1] in _del_keys:
            _dict.pop(key)


# @numba.njit(cache=True)
def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
    edges = []
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        targets.append(node)
        weights.append(weight)
        edges.append(edge)

    return edges, np.asarray(targets).ravel(), np.asarray(weights)


# @numba.njit(cache=True)
def calc_ppr_topk(indptr, indices, deg, alpha, epsilon, nodes, topk):
    edges = []
    targets = []
    weights = []
    del_nodes = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)
        # weight_np小到大排序
        idx_sort = np.argsort(weight_np)
        idx_topk = idx_sort[-topk:] # 取倒序topk个， 最重要
        idx_topk_rest = idx_sort[:-topk]
        # idx_topk = idx_sort[:topk] # 取顺序topk个，最不重要
        # idx_topk_rest = idx_sort[topk:]
        targets.append(node_np[idx_topk])
        weights.append(weight_np[idx_topk])
        del_nodes.append(node_np[idx_topk_rest])
        edges.append(edge)

    return edges, targets, weights, del_nodes


# @numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
    edges = [{}] * len(nodes)
    targets = [np.zeros(0, dtype=np.int64)] * len(nodes)
    weights = [np.zeros(0, dtype=np.float32)] * len(nodes)
    del_nodes = [np.zeros(0, dtype=np.int64)] * len(nodes)
    for i in numba.prange(len(nodes)):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)
        idx_sort = np.argsort(weight_np)
        idx_topk = idx_sort[-topk:]
        idx_topk_rest = idx_sort[:-topk]
        # idx_topk = idx_sort[:topk]
        # idx_topk_rest = idx_sort[topk:]
        targets[i] = node_np[idx_topk]
        weights[i] = weight_np[idx_topk]
        del_nodes[i] = node_np[idx_topk_rest]
        edges[i] = edge

    return edges, targets, weights, del_nodes




