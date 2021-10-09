import types

import numba
import numpy as np
from numba import int64

from graphgallery import functional as gf
from numba.typed import Dict
from numba.core import types


class PPRer:
    def __init__(self, adj_matrix, labels):
        self.adj_matrix = adj_matrix
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.out_degree = np.sum(adj_matrix > 0, axis=1).A1
        self.labels = labels

    # alpha >> 1 pay more attention to immediate neighbors
    # alpha >> 0 pay more attention to multi-hop neighbors
    def ppr_sample(self, targets, alpha, epsilon):
        edges, nodes, weights = calc_ppr(self.indptr, self.indices, self.out_degree, alpha, epsilon, np.asarray(targets))
        return edges, nodes

    def ppr_topk_sample(self, targets, alpha, epsilon, topk):
        edges, nodes, weights = calc_ppr_topk(self.indptr, self.indices, self.out_degree, alpha, epsilon, np.asarray(targets), topk)
        # delete_edges(edges, del_nodes)
        # _edges = update_edges(edges)
        return edges, nodes


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
                    if (vnode, unode) not in edges:
                        edges[(unode, vnode)] = 1

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


def dict_filter_key(_dict, _del_keys):
    _fdict = {}
    keys = list(_dict.keys())
    for key in keys:
        if key[0] not in _del_keys and key[1] not in _del_keys:
            _fdict[key] = 1
    return _fdict


def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
    edges = {}
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        targets.append(node)
        weights.append(weight)
        edges.update(edge)

    return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets).ravel(), np.asarray(weights)


def calc_ppr_topk(indptr, indices, deg, alpha, epsilon, nodes, topk):
    edges = {}
    targets = []
    weights = []
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
        # dict_del_key(edge, node_np[idx_topk_rest])
        # edges.update(edge)
        edges.update(dict_filter_key(edge, node_np[idx_topk_rest]))

    return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets).ravel(), np.asarray(weights)



