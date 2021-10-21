import numba
import numpy as np

from graphgallery import functional as gf
from utils import get_wl, get_wl_matrix, get_cross_entropy_matrix


class PPRer:
    def __init__(self, subgraph_type, adj_matrix, labels, alpha=0.25, logits=None, eps=1e-4):
        self.adj_matrix = adj_matrix
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.out_degree = np.sum(adj_matrix > 0, axis=1).A1
        self.labels = labels
        self.subgraph_type = subgraph_type

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
    # alpha >> 0 pay more attention to multi-hop neighbors， 节点也更多
    # 原始ppr，无采样数量限制
    def ppr_sample(self, targets):
        edges, nodes, weights = calc_ppr(self.indptr, self.indices, self.out_degree, self.alpha, self.eps, np.asarray(targets))
        return edges, nodes

    # ppr在采样的时候进行采样数量限制
    def ppr_nums_sample(self, targets, sample_nums):
        edges, nodes, weights = calc_ppr_nums(self.indptr, self.indices, self.out_degree, self.alpha, self.eps,
                                         np.asarray(targets), sample_nums)
        return edges, nodes

    # 原始ppr后，再根据weight排序，采样sample_nums(topk)个节点
    def ppr_topk_sample(self, targets, topk, descending):
        edges, nodes, weights = calc_ppr_topk(self.indptr, self.indices, self.out_degree, self.alpha, self.eps, np.asarray(targets), topk, descending)
        return edges, nodes

    # 原始ppr后，再根据wl，缩减候选集，无采样数量限制
    def ppr_wl_limit_sample(self, targets):
        edges, nodes, weights = calc_ppr_wl_limit(self.indptr, self.indices, self.out_degree, self.alpha, self.eps, np.asarray(targets), self.labels, self.wrong_label, self.wl)
        return edges, nodes

    # ppr在采样的时候进行采样数量限制+wl阈值，缩减候选集
    def ppr_wl_limit_nums_sample(self, targets, sample_nums):
        edges, nodes, weights = calc_ppr_wl_limit_nums(self.indptr, self.indices, self.out_degree, self.alpha, self.eps,
                                            np.asarray(targets), self.labels, self.wrong_label, self.wl, sample_nums)
        return edges, nodes

    # ppr公式乘wl，无采样数量限制 传wl 或 wl_cnt
    def ppr_wl_sample(self, targets):
        edges, nodes, weights = calc_ppr_wl(self.indptr, self.indices, self.out_degree, self.alpha, self.eps, np.asarray(targets), self.wl_cnt)
        return edges, nodes

    # ppr公式乘wl，wl阈值，缩减候选集，无采样数量限制
    def ppr_wl_limit_wl_sample(self, targets):
        edges, nodes, weights = calc_ppr_wl_limit_wl(self.indptr, self.indices, self.out_degree, self.alpha, self.eps,
                                            np.asarray(targets), self.labels, self.wrong_label, self.wl, self.wl_cnt)
        return edges, nodes

    # 原始ppr+topk+wl偏好
    def ppr_wl_topk_sample(self, targets, topk, descending):
        edges, nodes, weights = calc_ppr_wl_topk(self.indptr, self.indices, self.out_degree, self.alpha, self.eps,
                                              np.asarray(targets), topk, self.wl, descending)
        return edges, nodes


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node_nums(inode, indptr, indices, deg, alpha, epsilon, sample_nums):
    edges = {}
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0 and len(p) < sample_nums:
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


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node_wl(wl, inode, indptr, indices, deg, alpha, epsilon):
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
            _val = (1 - alpha) * res / deg[unode] * wl[unode]
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


# 原始ppr
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


# 原始ppr+限制数量
def calc_ppr_nums(indptr, indices, deg, alpha, epsilon, nodes, sample_nums):
    edges = {}
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node_nums(node, indptr, indices, deg, alpha, epsilon, sample_nums)
        targets.append(node)
        weights.append(weight)
        edges.update(edge)

    return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets).ravel(), np.asarray(weights)


# 原始ppr+公式乘上wl
def calc_ppr_wl(indptr, indices, deg, alpha, epsilon, nodes, wl):
    edges = {}
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node_wl(wl, node, indptr, indices, deg, alpha, epsilon)
        targets.append(node)
        weights.append(weight)
        edges.update(edge)

    return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets).ravel(), np.asarray(weights)


# 原始ppr+topk
def calc_ppr_topk(indptr, indices, deg, alpha, epsilon, nodes, topk, descending=False):
    edges = {}
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)

        # topk大于提取节点数量，退化成calc_ppr
        if len(node) <= topk:
            print('calc_ppr_topk back to calc_ppr')
            targets.append(node_np)
            weights.append(weight_np)
            edges.update(edge)
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
        # dict_del_key(edge, node_np[idx_topk_rest])
        # edges.update(edge)
        edges.update(dict_filter_key(edge, node_np[idx_topk_rest]))

    return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets).ravel(), np.asarray(weights)


# 原始ppr+topk+wl偏好
def calc_ppr_wl_topk(indptr, indices, deg, alpha, epsilon, nodes, topk, wl, descending=False):
    edges = {}
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)
        print(len(node_np))
        nodes_wl = wl[node_np]
        idx_wl = nodes_wl >= 0.5
        if idx_wl.sum() >= topk / 2:
            edge = dict_filter_key(edge, node_np[~idx_wl])
            node_np = node_np[idx_wl]
            weight_np = weight_np[idx_wl]
        print(len(node_np))

        # topk大于提取节点数量，退化成calc_ppr
        if len(node_np) <= topk:
            print('calc_ppr_topk back to calc_ppr')
            targets.append(node_np)
            weights.append(weight_np)
            edges.update(edge)
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
        edges.update(dict_filter_key(edge, node_np[idx_topk_rest]))

    return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets).ravel(), np.asarray(weights)


# 原始ppr+wl阈值
def calc_ppr_wl_limit(indptr, indices, deg, alpha, epsilon, nodes, labels, wrong_label, wl):
    edges = {}
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)
        idx_wl = labels[node_np] == wrong_label
        idx_one_wl = wl[node_np] >= 0.5
        idx = idx_wl | idx_one_wl
        if any(idx):
            targets.append(node_np[idx])
            weights.append(weight_np[idx])
            edges.update(dict_filter_key(edge, node_np[~idx]))
        else:
            print('iter:{}, node:{}, calc_wl_ppr no wrong label nodes'.format(i, node))
            targets.append(node)
            weights.append(weight)
            edges.update(edge)

    return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets).ravel(), np.asarray(weights)


# 原始ppr+wl阈值
def calc_ppr_wl_limit_nums(indptr, indices, deg, alpha, epsilon, nodes, labels, wrong_label, wl, sample_nums):
    edges = {}
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node_nums(node, indptr, indices, deg, alpha, epsilon, sample_nums)
        node_np, weight_np = np.array(node), np.array(weight)
        idx_wl = labels[node_np] == wrong_label
        idx_one_wl = wl[node_np] >= 0.5
        idx = idx_wl | idx_one_wl
        if any(idx):
            targets.append(node_np[idx])
            weights.append(weight_np[idx])
            edges.update(dict_filter_key(edge, node_np[~idx]))
        else:
            print('iter:{}, node:{}, calc_wl_ppr no wrong label nodes'.format(i, node))
            targets.append(node)
            weights.append(weight)
            edges.update(edge)

    return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets).ravel(), np.asarray(weights)


# ppr公式乘wl+wl阈值，缩减候选集，没有考虑采样数量上限
def calc_ppr_wl_limit_wl(indptr, indices, deg, alpha, epsilon, nodes, labels, wrong_label, wl, wl_cnt):
    edges = {}
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node_wl(wl_cnt, node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)
        idx_wl = labels[node_np] == wrong_label
        idx_one_wl = wl[node_np] >= 0.5
        idx = idx_wl | idx_one_wl
        if any(idx):
            targets.append(node_np[idx])
            weights.append(weight_np[idx])
            edges.update(dict_filter_key(edge, node_np[~idx]))
        else:
            print('iter:{}, node:{}, calc_wl_ppr no wrong label nodes'.format(i, node))
            targets.append(node)
            weights.append(weight)
            edges.update(edge)

    return gf.asedge(list(edges.keys()), shape='row_wise'), np.asarray(targets).ravel(), np.asarray(weights)


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