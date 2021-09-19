import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import random

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.targeted import PyTorch
from graphgallery.attack.targeted.targeted_attacker import TargetedAttacker
from graphgallery.datasets import NPZDataset


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
        self.loss_fn = nn.CrossEntropyLoss()
        random.seed(self.seed)
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
               subgraph_type='gcn',
               p=0.5):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)
        self.p = p
        self.added_edges = []
        self.non_added_edges = []
        if logit is None:
            logit = self.logits[target]
        idx = list(set(range(logit.size)) - set([self.target_label]))
        # wrong_label是次大概率的label
        wrong_label = idx[logit[idx].argmax()]
        self.wrong_label = torch.LongTensor([wrong_label]).to(self.device)
        self.true_label = torch.LongTensor([self.target_label]).to(self.device)
        self.subgraph_preprocessing(subgraph_type, attacker_nodes)
        offset = self.edge_weights.shape[0]
        for it in range(self.num_budgets):
            #         for it in tqdm(range(self.num_budgets),
            #                        desc='Peturbing Graph',
            #                        disable=disable):
            edge_grad, non_edge_grad = self.compute_gradient()

            with torch.no_grad():  # 不计算梯度
                edge_grad *= (-2 * self.edge_weights + 1)
                non_edge_grad *= (-2 * self.non_edge_weights + 1)
                gradients = torch.cat([edge_grad, non_edge_grad], dim=0)

            index = torch.argmax(gradients)
            if index < offset:
                u, v = self.edge_index[:, index]
                add = False
            else:
                index -= offset
                u, v = self.non_edge_index[:, index]
                add = True
            assert not self.is_modified(u, v)
            self.adj_flips[(u, v)] = it
            self.update_subgraph(u, v, index, add=add)
            if add:
                self.added_edges.append((u, v))
            else:
                self.non_added_edges.append((u, v))
        return self

    def subgraph_preprocessing(self, subgraph_type, attacker_nodes=None):
        target = self.target
        wrong_label = self.wrong_label
        neighbors = self.graph.adj_matrix[target].indices  # target的邻居id
        wrong_label_nodes = self.similar_nodes[wrong_label]  # 获取标签为wrong_label的节点
        if subgraph_type == 'gcn':
            sub_edges, sub_nodes = self.ego_subgraph()
        elif subgraph_type == 'spread_random':
            sub_edges, sub_nodes = self.ego_subgraph_spread_random()
        sub_edges = sub_edges.T  # shape [2, M]
        print(sub_edges.shape)

        if self.direct_attack or attacker_nodes is not None:
            influence_nodes = [target]
            wrong_label_nodes = np.setdiff1d(wrong_label_nodes, neighbors)
        else:
            influence_nodes = neighbors

        self.construct_sub_adj(influence_nodes, wrong_label_nodes, sub_nodes, sub_edges)

        if attacker_nodes is not None:
            if self.direct_attack:
                influence_nodes = [target]
                wrong_label_nodes = self.top_k_wrong_labels_nodes(
                    k=self.num_budgets + 1)

            else:
                influence_nodes = neighbors
                wrong_label_nodes = self.top_k_wrong_labels_nodes(
                    k=attacker_nodes)

            self.construct_sub_adj(influence_nodes, wrong_label_nodes,
                                   sub_nodes, sub_edges)

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

    # 返回值(edges, nodes)
    #    edges: shape [2, M], the edges of the subgraph
    #    nodes: shape [N], the nodes of the subgraph
    def ego_subgraph(self):
        return gf.ego_graph(self.graph.adj_matrix, self.target, self.K)

    def ego_subgraph_spread_random(self):
        return ego_subgraph_spread_random(self.graph.adj_matrix, self.target, self.p)

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

        nodes = np.union1d(sub_nodes, wrong_label_nodes)
        edge_weights = np.ones(sub_edges.shape[1], dtype=self.floatx)
        non_edge_weights = np.zeros(non_edges.shape[1], dtype=self.floatx)
        self_loop_weights = np.ones(nodes.shape[0], dtype=self.floatx)
        self_loop = np.row_stack([nodes, nodes])

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


def test(target=1, subgraph_type='gcn', seed=124, p=0.5):
    verbose = 0
    ################### Surrogate model ############################
    trainer = gg.gallery.nodeclas.SGC(seed=1000).setup_graph(graph, K=2).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=verbose,
                      epochs=100)

    ################### Attacker model ############################
    attacker = SGA(graph, seed=seed).process(trainer)
    attacker.attack(target, subgraph_type=subgraph_type, p=p)
    ################### Victim model ############################
    # Before attack
    trainer = gg.gallery.nodeclas.GCN(seed=seed).setup_graph(graph).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=verbose,
                      epochs=100)
    original_predict = trainer.predict(target, transform="softmax")

    # After attack
    trainer = gg.gallery.nodeclas.GCN(seed=seed).setup_graph(attacker.g).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=verbose,
                      epochs=100)
    perturbed_predict = trainer.predict(target, transform="softmax")

    ################### Results ############################
    # print("original prediction", original_predict)
    # print("perturbed prediction", perturbed_predict)
    target_label = graph.node_label[target]
    diff = original_predict[target_label] - perturbed_predict[target_label]
    return diff, attacker
    # print(f"The True label of node {target} is {target_label}.")



gg.set_backend("th")

data = NPZDataset('cora',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")

graph = data.graph
splits = data.split_nodes(random_state=15)
seeds = list(range(9999))

times = 3
used_seeds = random.sample(seeds, times)
used_seeds = seeds[:100]
us = []
atks = []
for seed in used_seeds:
    print('----------', seed)
    try:
        diff1, model1 = test(seed=seed)
        diff2, model2 = test(subgraph_type='spread_random', seed=seed, p=0.7)
    except AssertionError as e:
        print(e)
        continue
    except PermissionError as e:
        print(e)
        continue
    atks.append([diff1, diff2, diff2 - diff1])
    us.append(seed)
    if atks[-1][2] > 0:
        print('##########')
        print('[SGA_ORI, SGA_RD]:', atks)
        print('[Random seeds]:', us)
    print('##########\nSGA_ORI:')
    print('加边：', model1.added_edges)
    print('减边', model1.non_added_edges)
    print('##########')
    print('SGA_RD:')
    print('加边：', model2.added_edges)
    print('减边', model2.non_added_edges)
