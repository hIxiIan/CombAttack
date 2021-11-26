import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from torch import norm, sigmoid, pow
import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.targeted import TensorFlow
from graphgallery.attack.targeted.targeted_attacker import TargetedAttacker
from utils import get_wrong_rate, get_hop_neighbors, get_hop_rate


@TensorFlow.register()
class SGA(TargetedAttacker):
    def process(self, surrogate, reset=True):
        assert isinstance(surrogate, gg.gallery.nodeclas.SGC), surrogate

        K = surrogate.cfg.data.K  # NOTE: Be compatible with graphgallery
        # nodes with the same class labels
        self.similar_nodes = [
            np.where(self.graph.node_label == c)[0]
            for c in range(self.num_classes)
        ]

        with tf.device(self.device):
            W, b = surrogate.model.weights
            X = tf.convert_to_tensor(self.graph.node_attr,
                                     dtype=self.floatx)
            self.b = b
            self.XW = X @ W
            self.K = K
            self.logits = surrogate.predict(np.arange(self.num_nodes))
            self.loss_fn = sparse_categorical_crossentropy
            self.shape = self.graph.adj_matrix.shape
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        # for the added self-loop
        self.selfloop_degree = (self.degree + 1.).astype(self.floatx)
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
        self.target_original = gf.astensor(self.graph.adj_matrix[self.target].toarray())

        if logit is None:
            logit = self.logits[target]
        idx = list(set(range(logit.size)) - set([self.target_label]))
        wrong_label = idx[logit[idx].argmax()]
        if w_label is not None:
            wrong_label = w_label
        with tf.device(self.device):
            self.wrong_label = wrong_label
            self.true_label = tf.convert_to_tensor(self.target_label,
                                                   dtype=self.floatx)
            self.subgraph_preprocessing(attacker_nodes)
            offset = self.edge_weights.shape[0]

            # for indirect attack, the edges related to targeted node should not be considered
            if not direct_attack:
                row, col = self.edge_index
                mask = tf.convert_to_tensor(np.logical_and(row != target, col != target), dtype=gg.floatx())
            else:
                mask = 1.0

            for it in range(self.num_budgets):
                edge_grad, non_edge_grad = self.compute_gradient()
                edge_grad = normalize_GCN(self.edge_index, edge_grad,
                                          self.selfloop_degree)
                non_edge_grad = normalize_GCN(self.non_edge_index,
                                              non_edge_grad,
                                              self.selfloop_degree)
                edge_grad *= (-2 * self.edge_weights + 1) * mask
                non_edge_grad *= (-2 * self.non_edge_weights + 1)
                gradients = tf.concat([edge_grad, non_edge_grad], axis=0)
                index = tf.argmax(gradients)
                if index < offset:
                    u, v = self.edge_index[:, index]
                    if self.verbose_us:
                        print('iter:{}, max gradient:{}, delete edge:({}, {})'.format(it, gradients[index], u, v))
                    add = False
                else:
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
        self._wrong_ratio, self._wrong_length = get_wrong_rate(sub_nodes, wrong_label_nodes)
        self._sub_edges = sub_edges
        self._sub_nodes = sub_nodes
        self._neighbors = neighbors
        if self.direct_attack or attacker_nodes is not None:
            influence_nodes = [target]
            wrong_label_nodes = np.setdiff1d(wrong_label_nodes, neighbors)
        else:
            influence_nodes = neighbors

        hop_nodes, _ = get_hop_neighbors(self.graph.adj_matrix.indices, self.graph.adj_matrix.indptr, self.target)
        self._hop_ratio, self._hop_length, self._walk_length = get_hop_rate(sub_nodes, hop_nodes)

        self.construct_sub_adj(influence_nodes, wrong_label_nodes, sub_nodes, sub_edges)

        if attacker_nodes is not None:
            if self.direct_attack:
                influence_nodes = [target]
                wrong_label_nodes = self.top_k_wrong_labels_nodes(
                    k=self.num_budgets)

            else:
                influence_nodes = neighbors
                wrong_label_nodes = self.top_k_wrong_labels_nodes(
                    k=attacker_nodes)

            self.construct_sub_adj(influence_nodes, wrong_label_nodes,
                                   sub_nodes, sub_edges)
        if self.verbose_us:
            print('sub_edges:', self._sub_edges.shape)
            print('sub_non_edges:', self._sub_non_edges.shape)
            print('sub_nodes:', self._sub_nodes.shape)
            print('sub_nodes:', self._sub_nodes)
            print('sub_edges:', self._sub_edges)
            print('sub_non_edges:', self._sub_non_edges)

    @tf.function
    def SGC_conv(self, XW, adj):
        out = XW
        for _ in range(self.K):
            out = tf.sparse.sparse_dense_matmul(adj, out)
        return out

    def compute_gradient(self, eps=5.0):
        # edge_weights = self.edge_weights
        # non_edge_weights = self.non_edge_weights
        # self_loop_weights = self.self_loop_weights

        edge_weights = normalize_GCN(self.edge_index, self.edge_weights,
                                     self.selfloop_degree)
        non_edge_weights = normalize_GCN(self.non_edge_index,
                                         self.non_edge_weights,
                                         self.selfloop_degree)
        self_loop_weights = normalize_GCN(self.self_loop,
                                          self.self_loop_weights,
                                          self.selfloop_degree)

        with tf.GradientTape() as tape:
            tape.watch([edge_weights, non_edge_weights])

            weights = tf.concat([
                edge_weights, edge_weights, non_edge_weights, non_edge_weights,
                self_loop_weights
            ], axis=0)
            # weights = normalize_GCN(self.indices, weights, self.selfloop_degree)
            adj = tf.sparse.SparseTensor(self.indices.T, weights, self.shape)

            output = self.SGC_conv(self.XW, adj)
            logit = output[self.target] + self.b
            # model calibration
            logit = tf.nn.softmax(logit / eps)
            # cross-entropy loss
            loss = self.loss_fn(self.true_label, logit) - self.loss_fn(self.wrong_label, logit)

        edge_grad, non_edge_grad = tape.gradient(loss, [edge_weights, non_edge_weights])
        return edge_grad, non_edge_grad

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
        self._sub_non_edges = non_edges
        nodes = np.union1d(sub_nodes, wrong_label_nodes)
        self._sub_nodes = nodes
        edge_weights = np.ones(sub_edges.shape[1], dtype=self.floatx)
        non_edge_weights = np.zeros(non_edges.shape[1],
                                    dtype=self.floatx)
        self_loop_weights = np.ones(nodes.shape[0], dtype=self.floatx)
        self_loop = np.row_stack([nodes, nodes])

        self.indices = np.hstack([
            sub_edges, sub_edges[[1, 0]], non_edges,
            non_edges[[1, 0]], self_loop
        ])
        if self.blockchain:
            indices = np.hstack([
                non_edges,
                non_edges[[1, 0]], self_loop
            ])
            edge_weights = np.ones(0, dtype=self.floatx)
        with tf.device(self.device):
            self.edge_weights = tf.Variable(edge_weights, dtype=self.floatx)
            self.non_edge_weights = tf.Variable(non_edge_weights, dtype=self.floatx)
            self.self_loop_weights = tf.convert_to_tensor(self_loop_weights,
                                                          dtype=self.floatx)
        self.edge_index = sub_edges
        self.non_edge_index = non_edges
        self.self_loop = self_loop

    def top_k_wrong_labels_nodes(self, k):
        with tf.device(self.device):
            _, non_edge_grad = self.compute_gradient()
            _, index = tf.math.top_k(non_edge_grad, k=k, sorted=False)

        wrong_label_nodes = self.non_edge_index[1][index.numpy()]
        return wrong_label_nodes

    def update_subgraph(self, u, v, index, add=True):
        if add:
            self.non_edge_weights[index].assign(1.0)
            degree_delta = 1.0
        else:
            self.edge_weights[index].assign(0.0)
            degree_delta = -1.0

        self.selfloop_degree[u] += degree_delta
        self.selfloop_degree[v] += degree_delta


def normalize_GCN(indices, weights, degree):
    row, col = indices
    inv_degree = tf.pow(degree, -0.5)
    normed_weights = weights * tf.gather(inv_degree, row) * tf.gather(inv_degree, col)
    return normed_weights


@TensorFlow.register()
class SGAPD(SGA):
    def process(self, surrogate, reset=True):
        assert isinstance(surrogate, gg.gallery.nodeclas.SGCPDS), surrogate

        K = surrogate.cfg.data.K  # NOTE: Be compatible with graphgallery
        # nodes with the same class labels
        self.similar_nodes = [
            np.where(self.graph.node_label == c)[0]
            for c in range(self.num_classes)
        ]

        with tf.device(self.device):
            W, b = surrogate.model.weights
            X = tf.convert_to_tensor(self.graph.node_attr,
                                     dtype=self.floatx)
            self.b = b
            self.XW = X @ W
            self.K = K
            self.logits = surrogate.predict(np.arange(self.num_nodes))
            self.loss_fn = sparse_categorical_crossentropy
            self.shape = self.graph.adj_matrix.shape
        if reset:
            self.reset()
        return self

    def compute_gradient(self, eps=5.0):
        # edge_weights = self.edge_weights
        # non_edge_weights = self.non_edge_weights
        # self_loop_weights = self.self_loop_weights

        edge_weights = normalize_GCN(self.edge_index, self.edge_weights,
                                     self.selfloop_degree)
        non_edge_weights = normalize_GCN(self.non_edge_index,
                                         self.non_edge_weights,
                                         self.selfloop_degree)
        self_loop_weights = normalize_GCN(self.self_loop,
                                          self.self_loop_weights,
                                          self.selfloop_degree)

        with tf.GradientTape() as tape:
            tape.watch([edge_weights, non_edge_weights])

            weights = tf.concat([
                edge_weights, edge_weights, non_edge_weights, non_edge_weights,
                self_loop_weights
            ], axis=0)
            # weights = normalize_GCN(self.indices, weights, self.selfloop_degree)
            adj = tf.sparse.SparseTensor(self.indices.T, weights, self.shape)

            output = self.SGC_conv(self.XW, adj)
            # logit = output[self.target] + self.b
            # # model calibration
            # logit = tf.nn.softmax(logit / eps)
            # # cross-entropy loss
            # loss = self.loss_fn(self.true_label, logit) - self.loss_fn(self.wrong_label, logit)

            dim_n = output.shape[0]
            z = sigmoid(output[[self.target]].mm(output.t()))
            loss = - pow(norm(z - self.target_original, p='fro'), 2) / dim_n

        edge_grad, non_edge_grad = tape.gradient(loss, [edge_weights, non_edge_weights])
        return edge_grad, non_edge_grad
