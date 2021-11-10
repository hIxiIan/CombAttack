import torch
import random
from torch import Tensor
from torch import norm, sigmoid, pow
import torch.nn as nn
import numpy as np

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.targeted import PyTorch
from graphgallery.attack.targeted.targeted_attacker import TargetedAttacker
from copy import deepcopy
from utils import normalize_GCN, get_hop_neighbors, get_hop_rate, get_wrong_rate, to_array
from time import time

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
class SCA(TargetedAttacker):
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
               verbose_us=True,
               sampler=None):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)
        self.added_edges = []
        self.non_added_edges = []
        self.verbose_us = verbose_us
        self.sampler = sampler
        self.target_original = gf.astensor(self.graph.adj_matrix[self.target].toarray())
        if logit is None:
            logit = self.logits[target]
        idx = list(set(range(logit.size)) - set([self.target_label]))
        # wrong_label是次大概率的label
        wrong_label = idx[logit[idx].argmax()]

        # self.sampler = Sampler(self.graph.adj_matrix, self.graph.node_label, wrong_label, prob, p, q, self.seed, self.logits)
        self.wrong_label = torch.LongTensor([wrong_label]).to(self.device)
        self.true_label = torch.LongTensor([self.target_label]).to(self.device)
        if self.sampler.type_ == "cluster":
            self.subgraph_preprocessing_cluster()
        else:
            self.subgraph_preprocessing(attacker_nodes)
        offset = self.edge_weights.shape[0]

        # for indirect attack, the edges related to targeted node should not be considered
        if not direct_attack:
            row, col = self.edge_index
            mask = torch.FloatTensor(np.logical_and(row != target, col != target))
        else:
            mask = 1.0

        for it in range(self.num_budgets):
            #         for it in tqdm(range(self.num_budgets),
            #                        desc='Peturbing Graph',
            #                        disable=disable):
            edge_grad, non_edge_grad = self.compute_gradient()

            with torch.no_grad():  # 不计算梯度
                edge_grad *= (-2 * self.edge_weights + 1) * mask
                non_edge_grad *= (-2 * self.non_edge_weights + 1)
                gradients = torch.cat([edge_grad, non_edge_grad], dim=0) # 删边和减边的梯度cat成一维数组
            # 有可能选出来的边已经被攻击了，则选择次大的
            potential_times = len(gradients)
            while potential_times > 0:
                index = torch.argmax(gradients) # 求梯度最大的值的索引
                ori_index = index.item()
                if index < offset: # 该索引属于删边部分
                    u, v = self.edge_index[:, index] # 取节点
                    if self.verbose_us:
                        print('iter:{}, max gradient:{}, delete edge:({}, {})'.format(it, gradients[index], u, v))
                    add = False
                else: # 加边部分，直接将删边部分的索引offset减去
                    index -= offset
                    u, v = self.non_edge_index[:, index]
                    if self.verbose_us:
                        print('iter:{}, max gradient:{}, add edge:({}, {})'.format(it, gradients[index + offset], u, v))
                    add = True

                if self.is_modified(u, v):
                    gradients[ori_index] = 0.0
                    potential_times -= 1
                    continue
                else:
                    # print('potential_times:{}, total_times:{}'.format(potential_times, len(gradients)))
                    self.adj_flips[(u, v)] = it
                    self.update_subgraph(u, v, index, add=add)
                    if add:
                        self.added_edges.append((u, v))
                    else:
                        self.non_added_edges.append((u, v))
                    break
            # if potential_times == 0:
            #     assert False, 'all of the potential edges ({}) are modified, no more edges to attack'.format(len(gradients))
        return self

    def subgraph_preprocessing_cluster(self):
        sub_nodes = self.sampler.sub_nodes[self.sampler.targets_map[self.target]]
        deleted_edges = self.sampler.deleted_edges[self.sampler.targets_map[self.target]]
        added_edges = self.sampler.added_edges[self.sampler.targets_map[self.target]]

        wrong_label = self.wrong_label  # 分类概率次大的label
        wrong_label_nodes = self.similar_nodes[wrong_label]  # 获取标签为wrong_label的节点
        self._wrong_ratio, self._wrong_length = get_wrong_rate(sub_nodes, wrong_label_nodes)

        self._hop_ratio, self._hop_length, self._walk_length = 0, 0, 0
        self._sub_nodes = sub_nodes
        self._sub_edges = deleted_edges
        self._sub_non_edges = added_edges

        self.construct_sub_adj(sub_nodes, deleted_edges, added_edges)

        if self.verbose_us:
            print('sub_edges:', self._sub_edges.shape)
            print('sub_non_edges:', self._sub_non_edges.shape)
            print('sub_nodes:', self._sub_nodes.shape)
            print('sub_nodes:', self._sub_nodes)
            print('sub_edges:', self._sub_edges)
            print('sub_non_edges:', self._sub_non_edges)

    def subgraph_preprocessing(self, attacker_nodes=None):
        wrong_label = self.wrong_label # 分类概率次大的label
        wrong_label_nodes = self.similar_nodes[wrong_label]  # 获取标签为wrong_label的节点
        sub_edges = self.sampler.sample_edges[self.sampler.targets_map[self.target]]
        sub_nodes = self.sampler.sample_nodes[self.sampler.targets_map[self.target]]
        sub_edges = sub_edges.T  # shape [2, M]
        self._wrong_ratio, self._wrong_length = get_wrong_rate(sub_nodes, wrong_label_nodes)
        # 当提取的子图节点数量少于等于10个的时候，直接将wrong_label_nodes加入无连边集合

        non_edges = self.get_non_edges(sub_nodes)

        hop_nodes, _ = get_hop_neighbors(self.graph.adj_matrix.indices, self.graph.adj_matrix.indptr, self.target)
        # print(hop_nodes.shape, hop_nodes)
        # print(sub_nodes.shape, sub_nodes)
        self._hop_ratio, self._hop_length, self._walk_length = get_hop_rate(sub_nodes, hop_nodes)
        # print(self._hop_ratio, self._hop_length)
        self._sub_nodes = sub_nodes
        self._sub_edges = sub_edges
        self._sub_non_edges = non_edges

        # 构造子图，这一步是为了top_k_wrong_labels_nodes中计算梯度的时候有indices可用
        self.construct_sub_adj(sub_nodes, sub_edges, non_edges)

        if self.verbose_us:
            print('sub_edges:', self._sub_edges.shape)
            print('sub_non_edges:', self._sub_non_edges.shape)
            print('sub_nodes:', self._sub_nodes.shape)
            print('sub_nodes:', self._sub_nodes)
            print('sub_edges:', self._sub_edges)
            print('sub_non_edges:', self._sub_non_edges)

    def get_non_edges(self, sub_nodes, wrong_label_nodes=[]):
        target = self.target
        neighbors = self.graph.adj_matrix[target].indices  # target的邻居id
        # 直接攻击或者attacker_nodes不为None
        if self.direct_attack:
            influence_nodes = [target]  # 直接攻击，被影响的就是target
            target_neighbors = []
            target_neighbors.extend(influence_nodes)
            target_neighbors.extend(neighbors)
            non_nodes = np.setdiff1d(sub_nodes, target_neighbors)
        else:
            influence_nodes = neighbors  # 间接攻击，被影响的就是neighbors
            non_nodes = sub_nodes
            for infl in influence_nodes:
                infl_neighbors = self.graph.adj_matrix[infl].indices
                non_nodes = np.setdiff1d(non_nodes, infl_neighbors)
        if self.verbose_us:
            print('before non_nodes.shape:{}, non_nodes:{}'.format(non_nodes.shape, non_nodes))

        if len(wrong_label_nodes) > 0:
            wrong_label_nodes = np.setdiff1d(wrong_label_nodes, neighbors)
            if self.verbose_us:
                print('wrong_label_nodes.shape:{}, intersection:{}'.format(wrong_label_nodes.shape, np.intersect1d(non_nodes, wrong_label_nodes)))
            non_nodes = np.union1d(non_nodes, wrong_label_nodes)
        if self.verbose_us:
            print('after non_nodes.shape:{}, non_nodes:{}'.format(non_nodes.shape, non_nodes))

        length = len(non_nodes)
        non_edges = np.hstack([
            np.row_stack([np.tile(infl, length), non_nodes])
            for infl in influence_nodes
        ])
        return non_edges

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
        # 最小化loss，即true_label的概率越小，wrong_label的概率越大
        loss = self.loss_fn(logit, self.true_label) - self.loss_fn(logit, self.wrong_label)
        gradients = torch.autograd.grad(loss, [edge_weights, non_edge_weights], create_graph=False)
        return gradients

    def construct_sub_adj(self, sub_nodes, sub_edges, non_edges):
        edge_weights = np.ones(sub_edges.shape[1], dtype=self.floatx) # 边权重，初始化为1
        non_edge_weights = np.zeros(non_edges.shape[1], dtype=self.floatx)
        self_loop_weights = np.ones(sub_nodes.shape[0], dtype=self.floatx)
        self_loop = np.row_stack([sub_nodes, sub_nodes])

        # sub_edges, sub_edges[[1,0]]是方向相反的边
        if sub_edges.shape[1] == 0 or sub_edges.shape[0] == 0:
            indices = np.hstack([
                non_edges,
                non_edges[[1, 0]], self_loop
            ])
            edge_weights = np.ones(0, dtype=self.floatx)  # 边权重，初始化为1
        else:
            indices = np.hstack([
                sub_edges, sub_edges[[1, 0]], non_edges,
                non_edges[[1, 0]], self_loop
            ])

        self.indices = torch.LongTensor(indices).to(self.device)
        self.edge_weights = nn.Parameter(torch.tensor(edge_weights)).to(self.device)
        self.non_edge_weights = nn.Parameter(torch.tensor(non_edge_weights)).to(self.device)
        self.self_loop_weights = torch.tensor(self_loop_weights).to(self.device)

        self.edge_index = sub_edges
        self.non_edge_index = non_edges
        self.self_loop = self_loop

    def top_k_wrong_labels_nodes(self, k):
        _, non_edge_grad = self.compute_gradient()
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


@PyTorch.register()
class SCAPD(SCA):
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
        self.softmax_logits = surrogate.predict(np.arange(self.num_nodes), transform="softmax")
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