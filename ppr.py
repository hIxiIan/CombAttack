import numba
import numpy as np

from graphgallery import functional as gf
from utils import get_wl, get_wl_matrix, get_cross_entropy_matrix


class PPRer:
    def __init__(self, targets, sample_ratio, subgraph_type, adj_matrix, labels, alpha=0.25, ori_logits=None, logits=None, wl_limit=0.5, eps=1e-4):
        self.adj_matrix = adj_matrix
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.out_degree = np.sum(adj_matrix > 0, axis=1).A1
        self.labels = labels
        self.subgraph_type = subgraph_type
        self.ori_logits = ori_logits
        self.logits = logits

        self.wl_limit = wl_limit
        self.eps = eps
        self.alpha = alpha

        self.wrong_labels = None
        self.targets_map = {}
        self.sample_edges = []
        self.sample_nodes = []
        self.targets = np.asarray(targets)
        for i, target in enumerate(self.targets):
            self.targets_map[target] = i
        self.sample_nums = int(sample_ratio * adj_matrix.shape[0])
        self.wls = None
        self.wl_cnts = None

    def get_wrong_labels(self):
        wrong_labels = []
        for target in self.targets:
            logit = self.ori_logits[target]
            idx = list(set(range(logit.size)) - set([self.labels[target]]))
            wrong_label = idx[logit[idx].argmax()]
            wrong_labels.append(wrong_label)
        self.wrong_labels = np.array(wrong_labels)

    def get_all_wl(self):
        self.get_wrong_labels()
        wls = []
        wl_cnts = []
        for wrong_label in self.wrong_labels:
            wl, wl_cnt = get_wl(self.adj_matrix.indices, self.adj_matrix.indptr, self.labels, wrong_label,
                                          self.eps)
            wls.append(wl)
            wl_cnts.append(wl_cnt)
        self.wls = np.array(wls)
        self.wl_cnts = np.array(wl_cnts)

    def ppr_walk(self):
        if 'wl' in self.subgraph_type:
            self.get_all_wl()
            if self.subgraph_type == 'ppr_wl_limit':  # wl阈值
                sub_edges, sub_nodes = self.ppr_wl_limit_sample()
            elif self.subgraph_type == 'ppr_wl_limit_nums':
                sub_edges, sub_nodes = self.ppr_wl_limit_nums_sample()
            elif self.subgraph_type == 'ppr_wl':  # 公式乘wl
                sub_edges, sub_nodes = self.ppr_wl_sample()
            elif self.subgraph_type == 'ppr_wl_limit_wl':
                sub_edges, sub_nodes = self.ppr_wl_limit_wl_sample()
            elif self.subgraph_type == 'ppr_wl_topk_des':
                sub_edges, sub_nodes = self.ppr_wl_topk_sample(descending=True)
            elif self.subgraph_type == 'ppr_wl_topk_asc':
                sub_edges, sub_nodes = self.ppr_wl_topk_sample(descending=False)
            elif self.subgraph_type == 'ppr_wl_limit_topk_wl_asc':
                sub_edges, sub_nodes = self.ppr_wl_limit_topk_wl_sample(descending=False)
        else:
            if self.subgraph_type == 'ppr':
                sub_edges, sub_nodes = self.ppr_sample()
            elif self.subgraph_type == "ppr_nums":
                sub_edges, sub_nodes = self.ppr_nums_sample()
            elif self.subgraph_type == 'ppr_topk_des':
                sub_edges, sub_nodes = self.ppr_topk_sample(descending=True)
            elif self.subgraph_type == 'ppr_topk_asc':
                sub_edges, sub_nodes = self.ppr_topk_sample(descending=False)

        self.sample_edges = [gf.asedge(sub_edge, shape='row_wise') if len(sub_edge) > 0 else np.array([[],[]], dtype='int64') for sub_edge in sub_edges]
        self.sample_nodes = [np.unique(sub_node) for sub_node in sub_nodes]

    # alpha >> 1 pay more attention to immediate neighbors
    # alpha >> 0 pay more attention to multi-hop neighbors， 节点也更多
    # 原始ppr，无采样数量限制
    def ppr_sample(self):
        edges, nodes, _ = calc_ppr(self.indptr, self.indices, self.out_degree, self.alpha, self.eps, self.targets)
        return edges, nodes

    # ppr在采样的时候进行采样数量限制
    def ppr_nums_sample(self):
        edges, nodes, _ = calc_ppr_nums(self.indptr, self.indices, self.out_degree, self.alpha, self.eps,
                                         self.targets, self.sample_nums)
        return edges, nodes

    # 原始ppr后，再根据weight排序，采样sample_nums(topk)个节点
    def ppr_topk_sample(self, descending):
        edges, nodes, _ = calc_ppr_topk(self.indptr, self.indices, self.out_degree, self.alpha, self.eps, self.targets, self.sample_nums, descending)
        return edges, nodes

    # 原始ppr后，再根据wl，缩减候选集，无采样数量限制
    def ppr_wl_limit_sample(self):
        edges, nodes, _ = calc_ppr_wl_limit(self.indptr, self.indices, self.out_degree, self.alpha, self.eps, self.targets, self.labels, self.wrong_labels, self.wls)
        return edges, nodes

    # ppr在采样的时候进行采样数量限制+wl阈值，缩减候选集
    def ppr_wl_limit_nums_sample(self):
        edges, nodes, _ = calc_ppr_wl_limit_nums(self.indptr, self.indices, self.out_degree, self.alpha, self.eps,
                                            self.targets, self.labels, self.wrong_labels, self.wls, self.sample_nums)
        return edges, nodes

    # ppr公式乘wl，无采样数量限制 传wl 或 wl_cnt
    def ppr_wl_sample(self):
        edges, nodes, _ = calc_ppr_wl(self.indptr, self.indices, self.out_degree, self.alpha, self.eps, self.targets, self.wl_cnts)
        return edges, nodes

    # ppr公式乘wl，wl阈值，缩减候选集，无采样数量限制
    def ppr_wl_limit_wl_sample(self):
        edges, nodes, _ = calc_ppr_wl_limit_wl(self.indptr, self.indices, self.out_degree, self.alpha, self.eps,
                                            self.targets, self.labels, self.wrong_labels, self.wls, self.wl_cnts)
        return edges, nodes

    # 原始ppr+topk+wl偏好
    def ppr_wl_topk_sample(self, descending):
        edges, nodes, _ = calc_ppr_wl_topk(self.indptr, self.indices, self.out_degree, self.alpha, self.eps,
                                              self.targets, self.sample_nums, self.wls, descending)
        return edges, nodes

    # ppr公式乘wl+topk+wl阈值偏好
    def ppr_wl_limit_topk_wl_sample(self, descending):
        edges, nodes, _ = calc_ppr_wl_limit_wl_topk(self.indptr, self.indices, self.out_degree, self.alpha, self.eps,
                                              self.targets, self.sample_nums, self.wls, self.wl_cnts, descending)
        return edges, nodes


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32, 'unode': numba.int32, 'vnode': numba.int32})
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


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32, 'unode': numba.int32, 'vnode': numba.int32})
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


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32, 'unode': numba.int32, 'vnode': numba.int32})
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
@numba.jit(cache=True, nopython=True)
def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
    edges = []
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        targets.append(node)
        weights.append(weight)
        edges.append(list(edge.keys()))

    return edges, targets, weights


# 原始ppr+限制数量
@numba.jit(cache=True, nopython=True)
def calc_ppr_nums(indptr, indices, deg, alpha, epsilon, nodes, sample_nums):
    edges = []
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node_nums(node, indptr, indices, deg, alpha, epsilon, sample_nums)
        targets.append(node)
        weights.append(weight)
        edges.append(list(edge.keys()))

    return edges, targets, weights


# 原始ppr+公式乘上wl
@numba.jit(cache=True, nopython=True)
def calc_ppr_wl(indptr, indices, deg, alpha, epsilon, nodes, wls):
    edges = []
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node_wl(wls[i], node, indptr, indices, deg, alpha, epsilon)
        targets.append(node)
        weights.append(weight)
        edges.append(list(edge.keys()))

    return edges, targets, weights


# 原始ppr+topk
@numba.jit(cache=True, nopython=True)
def calc_ppr_topk(indptr, indices, deg, alpha, epsilon, nodes, topk, descending=False):
    edges = []
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)

        # topk大于提取节点数量，退化成calc_ppr
        if len(node) <= topk:
            targets.append(node_np)
            weights.append(weight_np)
            edges.append(list(edge.keys()))
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
        edge = dict_filter_key(np.array(list(edge.keys())), node_np[idx_topk_rest])
        edges.append(list(edge.keys()))

    return edges, targets, weights


# 原始ppr+topk+wl偏好
@numba.jit(cache=True, nopython=True)
def calc_ppr_wl_topk(indptr, indices, deg, alpha, epsilon, nodes, topk, wls, descending=False):
    edges = []
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)
        nodes_wl = wls[i][node_np]
        idx_wl = nodes_wl >= 0.5
        if idx_wl.sum() >= topk / 2:
            edge = dict_filter_key(np.array(list(edge.keys())), node_np[~idx_wl])
            node_np = node_np[idx_wl]
            weight_np = weight_np[idx_wl]

        # topk大于提取节点数量，退化成calc_ppr
        if len(node_np) <= topk:
            # print('calc_ppr_topk back to calc_ppr')
            targets.append(node_np)
            weights.append(weight_np)
            edges.append(list(edge.keys()))
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
        edge = dict_filter_key(np.array(list(edge.keys())), node_np[idx_topk_rest])
        edges.append(list(edge.keys()))

    return edges, targets, weights


# ppr公式乘wl+topk+wl偏好
@numba.jit(cache=True, nopython=True)
def calc_ppr_wl_limit_wl_topk(indptr, indices, deg, alpha, epsilon, nodes, topk, wls, wl_cnts, descending=False):
    edges = []
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node_wl(wl_cnts[i], node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)
        nodes_wl = wls[i][node_np]
        idx_wl = nodes_wl >= 0.5
        if idx_wl.sum() >= topk / 2:
            edge = dict_filter_key(np.array(list(edge.keys())), node_np[~idx_wl])
            node_np = node_np[idx_wl]
            weight_np = weight_np[idx_wl]

        # topk大于提取节点数量，退化成calc_ppr
        if len(node_np) <= topk:
            # print('calc_ppr_topk back to calc_ppr')
            targets.append(node_np)
            weights.append(weight_np)
            edges.append(list(edge.keys()))
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
        edge = dict_filter_key(np.array(list(edge.keys())), node_np[idx_topk_rest])
        edges.append(list(edge.keys()))

    return edges, targets, weights


# 原始ppr+wl阈值
@numba.jit(cache=True, nopython=True)
def calc_ppr_wl_limit(indptr, indices, deg, alpha, epsilon, nodes, labels, wrong_labels, wls):
    edges = []
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)
        idx_wl = labels[node_np] == wrong_labels[i]
        idx_one_wl = wls[i][node_np] >= 0.5
        idx = idx_wl | idx_one_wl
        if np.any(idx):
            targets.append(node_np[idx])
            weights.append(weight_np[idx])
            edge = dict_filter_key(np.array(list(edge.keys())), node_np[~idx])
            edges.append(list(edge.keys()))
        else:
            targets.append(node_np)
            weights.append(weight_np)
            edges.append(list(edge.keys()))

    return edges, targets, weights


# 原始ppr+wl阈值
@numba.jit(cache=True, nopython=True)
def calc_ppr_wl_limit_nums(indptr, indices, deg, alpha, epsilon, nodes, labels, wrong_labels, wls, sample_nums):
    edges = []
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node_nums(node, indptr, indices, deg, alpha, epsilon, sample_nums)
        node_np, weight_np = np.array(node), np.array(weight)
        idx_wl = labels[node_np] == wrong_labels[i]
        idx_one_wl = wls[i][node_np] >= 0.5
        idx = idx_wl | idx_one_wl
        if np.any(idx):
            targets.append(node_np[idx])
            weights.append(weight_np[idx])
            edge = dict_filter_key(np.array(list(edge.keys())), node_np[~idx])
            edges.append(list(edge.keys()))
        else:
            targets.append(node_np)
            weights.append(weight_np)
            edges.append(list(edge.keys()))

    return edges, targets, weights


# ppr公式乘wl+wl阈值，缩减候选集，没有考虑采样数量上限
@numba.jit(cache=True, nopython=True)
def calc_ppr_wl_limit_wl(indptr, indices, deg, alpha, epsilon, nodes, labels, wrong_labels, wls, wl_cnts):
    edges = []
    targets = []
    weights = []
    for i, node in enumerate(nodes):
        node, weight, edge = _calc_ppr_node_wl(wl_cnts[i], node, indptr, indices, deg, alpha, epsilon)
        node_np, weight_np = np.array(node), np.array(weight)
        idx_wl = labels[node_np] == wrong_labels[i]
        idx_one_wl = wls[i][node_np] >= 0.5
        idx = idx_wl | idx_one_wl
        if np.any(idx):
            targets.append(node_np[idx])
            weights.append(weight_np[idx])
            edge = dict_filter_key(np.array(list(edge.keys())), node_np[~idx])
            edges.append(list(edge.keys()))
        else:
            targets.append(node_np)
            weights.append(weight_np)
            edges.append(list(edge.keys()))

    return edges, targets, weights


@numba.jit(nopython=True)
def dict_filter_key(_dict_keys, _del_keys):
    _fdict = {}
    for key in _dict_keys:
        if key[0] not in _del_keys and key[1] not in _del_keys:
            _fdict[(key[0], key[1])] = 1
    return _fdict