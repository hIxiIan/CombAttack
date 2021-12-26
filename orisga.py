import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from torch import norm, sigmoid, pow
import graphgallery as gg
from utils import get_wrong_rate, get_hop_neighbors, get_hop_rate
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.targeted import PyTorch
from graphgallery.attack.targeted.targeted_attacker import TargetedAttacker


try:
    """It will be faster with torch_geometric"""
    from torch_geometric.typing import Adj, OptTensor
    from torch_sparse import SparseTensor, matmul
    from torch_geometric.nn.conv import MessagePassing

    class SGConv(MessagePassing):
        def __init__(self, K=2, **kwargs):
            kwargs.setdefault('aggr', 'add')
            super().__init__(**kwargs)
            self.K = K

        def forward(self, x: Tensor, edge_index: Adj,
                    edge_weight: OptTensor = None) -> Tensor:
            for _ in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
            return x

        def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
            return edge_weight.view(-1, 1) * x_j

        def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
            return matmul(adj_t, x, reduce=self.aggr)

        def __repr__(self):
            return '{}(K={})'.format(self.__class__.__name__, self.K)

except ImportError:
    class SGConv(nn.Module):
        def __init__(self, K=2):
            super().__init__()
            self.K = K

        def forward(self, x: Tensor, edge_index: Tensor,
                    edge_weight: Tensor) -> Tensor:
            N = x.size(0)
            adj = torch.sparse.FloatTensor(edge_index, edge_weight, (N, N))
            for _ in range(self.K):
                x = torch.sparse.mm(adj, x)
            return x

        def __repr__(self):
            return '{}(K={})'.format(self.__class__.__name__, self.K)


@PyTorch.register()
class SGA(TargetedAttacker):
    def process(self, surrogate, reset=True):
        assert isinstance(surrogate, gg.gallery.nodeclas.SGC), surrogate

        K = surrogate.cfg.data.K  # NOTE: Be compatible with graphgallery
        # nodes with the same class labels
        self.similar_nodes = [
            np.where(self.graph.node_label == c)[0]
            for c in range(self.num_classes)
        ]

        W, b = surrogate.model.parameters()
        W, b = W.to(self.device), b.to(self.device)
        X = torch.tensor(self.graph.node_attr).to(self.device)
        self.b = b
        self.XW = X @ W.T
        self.SGC = SGConv(K).to(self.device)
        self.K = K
        self.logits = surrogate.predict(np.arange(self.num_nodes))
        self.softmax_logits = surrogate.predict(np.arange(self.num_nodes), transform="softmax")
        self.loss_fn = nn.CrossEntropyLoss()
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        # for the added self-loop
        self.selfloop_degree = torch.tensor(self.degree + 1.).to(self.device)
        self.adj_flips = {}
        self.wrong_label = None
        return self

    def attack(self,
               target,
               num_budgets=None,
               logit=None,
               attacker_nodes=3,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               disable=False,
               w_label=None,
               verbose_us=True,
               blockchain=None):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)
        self.verbose_us = verbose_us
        self.added_edges = []
        self.non_added_edges = []
        self.with_w_label = False
        self.blockchain = blockchain
        self.target_original = gf.astensor(self.graph.adj_matrix[self.target].toarray()).to(self.device)
        if logit is None:
            logit = self.logits[target]
        idx = list(set(range(logit.size)) - set([self.target_label]))
        wrong_label = idx[logit[idx].argmax()]
        if w_label is not None:
            wrong_label = w_label
        self.wrong_label = torch.LongTensor([wrong_label]).to(self.device)
        self.true_label = torch.LongTensor([self.target_label]).to(self.device)
        self.subgraph_preprocessing(attacker_nodes)
        offset = self.edge_weights.shape[0]

        # for indirect attack, the edges related to targeted node should not be considered
        if not direct_attack:
            row, col = self.edge_index
            mask = torch.FloatTensor(np.logical_and(row != target, col != target)).to(self.device)
        else:
            mask = 1.0

        for it in range(self.num_budgets):
            edge_grad, non_edge_grad = self.compute_gradient()

            with torch.no_grad():
                edge_grad *= (-2 * self.edge_weights + 1) * mask
                non_edge_grad *= (-2 * self.non_edge_weights + 1)
                gradients = torch.cat([edge_grad, non_edge_grad], dim=0)

            index = torch.argmax(gradients)
            if index < offset:  # 该索引属于删边部分
                u, v = self.edge_index[:, index]  # 取节点
                if self.verbose_us:
                    print('iter:{}, max gradient:{}, delete edge:({}, {})'.format(it, gradients[index], u, v))
                add = False
            else:  # 加边部分，直接将删边部分的索引offset减去
                index -= offset
                u, v = self.non_edge_index[:, index]
                if self.verbose_us:
                    print('iter:{}, max gradient:{}, add edge:({}, {})'.format(it, gradients[index + offset], u, v))
                add = True
            assert not self.is_modified(u, v)
            self.adj_flips[(u, v)] = it
            self.update_subgraph(u, v, index, add=add)
            if add:
                self.added_edges.append((u, v))
            else:
                self.non_added_edges.append((u, v))
        return self

    def subgraph_preprocessing(self, attacker_nodes=None):
        target = self.target
        wrong_label = self.wrong_label
        neighbors = self.graph.adj_matrix[target].indices
        wrong_label_nodes = self.similar_nodes[wrong_label]
        sub_edges, sub_nodes = self.ego_subgraph()
        sub_edges = sub_edges.T  # shape [2, M]
        # self._wrong_ratio, self._wrong_length = get_wrong_rate(sub_nodes, wrong_label_nodes)
        # self._sub_edges = sub_edges
        # self._sub_nodes = sub_nodes
        # self._neighbors = neighbors
        # print(self.direct_attack)
        if self.direct_attack:
            # print('direct')
            influence_nodes = [target]
            wrong_label_nodes = np.setdiff1d(wrong_label_nodes, neighbors)
        else:
            # print('indirect')
            influence_nodes = neighbors
        # hop_nodes, _ = get_hop_neighbors(self.graph.adj_matrix.indices, self.graph.adj_matrix.indptr, self.target)
        # print(hop_nodes.shape, hop_nodes)
        # print(sub_nodes.shape, sub_nodes)
        # self._hop_ratio, self._hop_length, self._walk_length = get_hop_rate(sub_nodes, hop_nodes)
        self.construct_sub_adj(influence_nodes, wrong_label_nodes, sub_nodes, sub_edges)
        # if self.verbose_us:
        # print('sub_non_edges:', self._sub_non_edges.shape)

        if attacker_nodes is not None:
            if self.direct_attack:
                influence_nodes = [target]
                wrong_label_nodes = self.top_k_wrong_labels_nodes(
                    k=self.num_budgets + 1)
                # print(wrong_label_nodes)
            else:
                influence_nodes = neighbors
                wrong_label_nodes = self.top_k_wrong_labels_nodes(
                    k=attacker_nodes)
                # print(wrong_label_nodes)
            self.construct_sub_adj(influence_nodes, wrong_label_nodes,
                                   sub_nodes, sub_edges)
        # if self.verbose_us:
        #     print('sub_edges:', self._sub_edges.shape)
        #     print('sub_non_edges:', self._sub_non_edges.shape)
        #     print('sub_nodes:', self._sub_nodes.shape)
        #     print('sub_nodes:', self._sub_nodes)
        #     print('sub_edges:', self._sub_edges)
        #     print('sub_non_edges:', self._sub_non_edges)

    def compute_gradient(self, eps=5.0):

        edge_weights = self.edge_weights
        non_edge_weights = self.non_edge_weights
        self_loop_weights = self.self_loop_weights
        weights = torch.cat([
            edge_weights, edge_weights, non_edge_weights, non_edge_weights,
            self_loop_weights
        ], dim=0)

        weights = normalize_GCN(self.indices, weights, self.selfloop_degree)
        output = self.SGC(self.XW, self.indices, weights)

        logit = output[self.target] + self.b
        # model calibration
        logit = logit.view(1, -1) / eps
        loss = self.loss_fn(logit, self.true_label) - self.loss_fn(logit, self.wrong_label)  # nll_loss
        gradients = torch.autograd.grad(loss, [edge_weights, non_edge_weights], create_graph=False)
        return gradients

    def ego_subgraph(self):
        return gf.ego_graph(self.graph.adj_matrix, self.target, self.K)

    def construct_sub_adj(self, influence_nodes, wrong_label_nodes, sub_nodes,
                          sub_edges):
        length = len(wrong_label_nodes)
        non_edges = np.hstack([
            np.row_stack([np.tile(infl, length), wrong_label_nodes])
            for infl in influence_nodes
        ])

        if len(influence_nodes) > 1:
            # TODO: considering self-loops
            mask = self.graph.adj_matrix[non_edges[0],
                                         non_edges[1]].A1 == 0
            non_edges = non_edges[:, mask]
        # self._sub_non_edges = non_edges
        nodes = np.union1d(sub_nodes, wrong_label_nodes)
        # self._sub_nodes = nodes
        edge_weights = np.ones(sub_edges.shape[1], dtype=self.floatx)
        non_edge_weights = np.zeros(non_edges.shape[1], dtype=self.floatx)
        self_loop_weights = np.ones(nodes.shape[0], dtype=self.floatx)
        self_loop = np.row_stack([nodes, nodes])
        # print('influence_nodes', influence_nodes)
        # print('sub_nodes', nodes, len(nodes)) # 排序
        # print('deleted_edges', sub_edges)
        # print('added_edges', list(non_edges[0]))# 排序
        # print('added_edges', list(non_edges[1]))  # 排序
        # exit()
        indices = np.hstack([
            sub_edges, sub_edges[[1, 0]], non_edges,
            non_edges[[1, 0]], self_loop
        ])
        if self.blockchain:
            indices = np.hstack([
                non_edges,
                non_edges[[1, 0]], self_loop
            ])
            edge_weights = np.ones(0, dtype=self.floatx)

        self.indices = torch.LongTensor(indices).to(self.device)
        self.edge_weights = nn.Parameter(torch.tensor(edge_weights)).to(self.device)
        self.non_edge_weights = nn.Parameter(torch.tensor(non_edge_weights)).to(self.device)
        self.self_loop_weights = torch.tensor(self_loop_weights).to(self.device)

        self.edge_index = sub_edges
        self.non_edge_index = non_edges
        self.self_loop = self_loop

    def top_k_wrong_labels_nodes(self, k):
        _, non_edge_grad = self.compute_gradient()
        # print('non_edge_grad:{}, k:{}'.format(len(non_edge_grad), k))
        k = min(len(non_edge_grad), k)
        _, index = torch.topk(non_edge_grad, k=k, sorted=False)

        wrong_label_nodes = self.non_edge_index[1][index.cpu()]
        return wrong_label_nodes

    def update_subgraph(self, u, v, index, add=True):
        if add:
            self.non_edge_weights[index] = 1.0
            self.selfloop_degree[u] += 1
            self.selfloop_degree[v] += 1
        else:
            self.edge_weights[index] = 0.0
            self.selfloop_degree[u] -= 1
            self.selfloop_degree[v] -= 1


def normalize_GCN(indices, weights, degree):
    row, col = indices
    inv_degree = torch.pow(degree, -0.5)
    normed_weights = weights * inv_degree[row] * inv_degree[col]
    return normed_weights


@PyTorch.register()
class SGAPD(SGA):
    def process(self, surrogate, reset=True):
        assert isinstance(surrogate, gg.gallery.nodeclas.SGCPDS), surrogate

        K = surrogate.cfg.data.K  # NOTE: Be compatible with graphgallery
        # nodes with the same class labels
        self.similar_nodes = [
            np.where(self.graph.node_label == c)[0]
            for c in range(self.num_classes)
        ]

        W, b = surrogate.model.parameters()
        W, b = W.to(self.device), b.to(self.device)
        X = torch.tensor(self.graph.node_attr).to(self.device)
        self.b = b
        self.XW = X @ W.T
        self.SGC = SGConv(K).to(self.device)
        self.K = K
        self.logits = surrogate.predict(np.arange(self.num_nodes))
        # self.loss_fn = nn.CrossEntropyLoss()
        if reset:
            self.reset()
        return self

    def compute_gradient(self, eps=5.0):
        edge_weights = self.edge_weights
        non_edge_weights = self.non_edge_weights
        self_loop_weights = self.self_loop_weights
        weights = torch.cat([
            edge_weights, edge_weights, non_edge_weights, non_edge_weights,
            self_loop_weights
        ], dim=0)

        weights = normalize_GCN(self.indices, weights, self.selfloop_degree)
        output = self.SGC(self.XW, self.indices, weights)
        dim_n = output.shape[0]
        z = sigmoid(output[[self.target]].mm(output.t()))
        loss = - pow(norm(z - self.target_original, p='fro'), 2) / dim_n
        gradients = torch.autograd.grad(loss, [edge_weights, non_edge_weights], create_graph=False)
        return gradients
