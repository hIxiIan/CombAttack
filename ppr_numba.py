import numba
import numpy as np

from graphgallery import functional as gf
from utils import get_wl, get_wl_matrix, get_cross_entropy_matrix


class PPRer:
    def __init__(self, adj_matrix, labels, alpha=0.25, logits=None, eps=1e-4):
        self.adj_matrix = adj_matrix
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.out_degree = np.sum(adj_matrix > 0, axis=1).A1
        self.labels = labels

        if logits is not None:
            self.ce_matrix = get_cross_entropy_matrix(logits)

        self.wrong_label = None
        self.wl = None
        self.wl_cnt = None
        self.wl_cnt_matrix = None
        self.eps = eps
        self.alpha = alpha

    def set_wrong_label(self, wrong_label):
        self.wrong_label = wrong_label
        self.wl, self.wl_cnt = get_wl(self.adj_matrix.indices, self.adj_matrix.indptr, self.labels, wrong_label, self.eps)
        self.wl_cnt_matrix = get_wl_matrix(self.wl_cnt)

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

    # 原始ppr+topk+wl偏好
    def ppr_wl_topk_sample(self, targets, topk, descending):
        edges, nodes, weights, del_nodes = calc_ppr_wl_topk(self.indptr, self.indices, self.out_degree, self.alpha, self.eps,
                                                 np.asarray(targets), topk, self.wl, descending)
        delete_edges(edges, del_nodes)
        _edges = update_edges(edges)
        return gf.asedge(list(_edges.keys()), shape='row_wise'), np.asarray(nodes).ravel()


# 这里好像有个int32和int64的转化，numba有报警
@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
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


# 原始ppr+topk+wl偏好
@numba.njit(cache=True)
def calc_ppr_wl_topk(indptr, indices, deg, alpha, epsilon, nodes, topk, wl, descending=False):
    edges = []
    targets = []
    weights = []
    del_nodes = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)

        del_node = []
        nodes_wl = wl[node_np]
        idx_wl = nodes_wl >= 0.5
        if idx_wl.sum() >= topk / 2:
            del_node.extend(node_np[~idx_wl])
            node_np = node_np[idx_wl]
            weight_np = weight_np[idx_wl]

        # topk大于提取节点数量，退化成calc_ppr
        if len(node_np) <= topk:
            print('calc_ppr_topk back to calc_ppr')
            targets.append(node_np)
            weights.append(weight_np)
            edges.append(edge)
            continue
        # weight_np小到大排序
        idx_sort = np.argsort(weight_np)

        if descending:
            idx_topk = idx_sort[-topk:] # 取倒序topk个， 最重要
            idx_topk_rest = idx_sort[:-topk]
        else:
            idx_topk = idx_sort[:topk] # 取顺序topk个，最不重要
            idx_topk_rest = idx_sort[topk:]
        targets.append(node_np[idx_topk])
        weights.append(weight_np[idx_topk])
        edges.append(edge)
        del_node.extend(node_np[idx_topk_rest])
        del_nodes.append(del_node)

    return edges, targets, weights, del_nodes


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


