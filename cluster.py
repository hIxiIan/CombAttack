import graphgallery as gg
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from graphgallery import functional as gf
from sklearn.cluster import KMeans
from numba import njit
from sklearn.manifold import TSNE
from time import time
from utils import get_wrong_labels, mapCluster2GCN

EUCLIDEAN = "euclidean"
WRONG_LABELS = "wrong_labels"

class Cluster:
    def __init__(self, direct_attack, embed_type, targets, model, graph, sample_ratio, logits, softmax_logits, parms):
        self.direct_attack = direct_attack
        self.embed_type = embed_type
        self.targets = np.array(targets)
        self.model = model
        self.graph = graph
        self.indices = self.graph.adj_matrix.indices
        self.indptr = self.graph.adj_matrix.indptr
        self.n_nodes = np.array(range(graph.adj_matrix.shape[0]))
        self.n_classes = len(set(graph.node_label))
        self.sample_nums = int(sample_ratio * graph.adj_matrix.shape[0])
        self.logits = logits
        self.parms = parms

        self.sur_labels = np.array([lo.argmax() for lo in softmax_logits])
        self.wrong_labels = get_wrong_labels(logits, targets, graph.node_label)
        self.z = None
        self.tsne = None
        self.cluster_type = "KMeans"
        self.cluser_model = None
        self.cluster_label_pred = None
        self.cluster_centroidds = None
        self.intertia = None
        self.farthest_idx = None
        self.farthest_idx_dict = None

        self.sub_nodes = None
        self.deleted_edges = None
        self.added_edges = None

        self.targets_map = {}
        for i, target in enumerate(targets):
            self.targets_map[target] = i

        self.layerMap = {
            'GCN2': 'GCNConv',
            'SGC2': 'Linear',
            'MLP': 'Linear',
            'FastGCN': 'Linear_GCNConv'
        }
        self.actMap = {
            'GCN2': 'ReLU',
            'SGC2': 'Noop',
            'MLP': 'ReLU',
            'FastGCN': 'ReLU'
        }

        self.do()

    def get_layer_act_idx(self, layers):
        embed_type = self.embed_type
        layerMap = self.layerMap
        actMap = self.actMap

        layerIdx = {}
        actIdx = {}
        lmo = layerMap[embed_type]
        lms = [layerMap[embed_type]]
        ac = actMap[embed_type]
        if embed_type == "FastGCN":
            lms = lms[0].split('_')
        for i, ly in enumerate(str(layers).split('\n')[1:-1]):
            for lm in lms:
                if lm in ly:
                    if lmo not in layerIdx:
                        layerIdx[lmo] = [i]
                    else:
                        layerIdx[lmo].append(i)
            if ac in ly:
                if ac not in actIdx:
                    actIdx[ac] = [i]
                else:
                    actIdx[ac].append(i)
        return layerIdx, actIdx

    def get_conv_idx(self, layer):
        lay_act = self.parms.lay_act
        lay_act_cnt = self.parms.lay_act_cnt
        layerIdx, actIdx = self.get_layer_act_idx(layer)
        lm = self.layerMap[self.embed_type]
        ac = self.actMap[self.embed_type]
        if lay_act == "layer":
            lay_act_cnt = max(lay_act_cnt, 1)
            lay_act_cnt = min(lay_act_cnt, len(layerIdx[lm]))
            return layerIdx[lm][lay_act_cnt - 1]
        lay_act_cnt = max(lay_act_cnt, 1)
        lay_act_cnt = min(lay_act_cnt, len(actIdx[ac]))
        return actIdx[ac][lay_act_cnt - 1]

    @torch.no_grad()
    def get_predict(self):
        if self.parms.lay_act_cnt > 100:
            self.z = self.model.predict(self.n_nodes)
        elif self.embed_type in ["GCN", "GCN2", "GAT", "FastGCN"]:
            conv = self.model.model.conv
            conv = conv[:self.get_conv_idx(conv) + 1]
            print(conv)
            self.z = conv(self.model.cache.X, self.model.cache.A).cpu().numpy()
        elif "MLP" in self.embed_type or "SGC2" in self.embed_type:
            lin = self.model.model.lin
            lin = lin[:self.get_conv_idx(lin) + 1]
            print(lin)
            self.z = lin(self.model.cache.X).cpu().numpy()
        elif "PPNP" in self.embed_type:
            lin = self.model.model.lin[:-3]
            propagation = self.model.model.propagation
            x = lin(self.model.cache.X)
            self.z = propagation(x, self.model.cache.A).cpu().numpy()
        elif self.embed_type in ["DW", 'N2V', 'BANE']:
            self.z = self.model.get_embedding()
        elif self.embed_type == "ClusterGCN":
            conv = self.model.model.conv[:-3]
            nums_cluster = len(self.model.cache.cluster_member)
            z = []
            z_idx = []
            for cluster in range(nums_cluster):
                tmp_z = conv(self.model.cache.batch_x[cluster], self.model.cache.batch_adj[cluster]).cpu().numpy()
                z.extend(tmp_z)
                z_idx.extend(self.model.cache.cluster_member[cluster])
            z = np.array(z)
            z_idx = np.array(z_idx)
            idx_ = np.argsort(z_idx)
            self.z = z[idx_]
        elif self.embed_type in ["GraphMLP"]:
            self.z = self.model.model.mlp(self.model.cache.X).cpu().numpy()
        else:
            self.z = self.model.predict(self.n_nodes)
        print('z shape: ', self.z.shape)

    def do(self):
        start = time()
        self.get_predict()
        self.do_cluster()
        self.get_candidates()
        # self.visualization()
        end = time()
        self.cluster_cost_time = end - start

    def compute_distance(self, embed1, embed2):
        return pow(embed1 - embed2, 2).sum()

    def get_farthest_idx(self):
        if self.parms.distance_type == EUCLIDEAN:
            farthest_idx = np.zeros((self.n_classes, self.n_classes))
            farthest = np.zeros((self.n_classes, self.n_classes))
            for i in range(self.n_classes):
                for j in range(i + 1, self.n_classes):
                    distance = self.compute_distance(self.cluster_centroidds[i], self.cluster_centroidds[j])
                    farthest[i][j] = farthest[j][i] = distance
            df = pd.DataFrame(farthest)
            for i in range(self.n_classes):
                farthest_idx[i] = np.array(df.iloc[i].sort_values(ascending=False).index)
            self.farthest_idx = farthest_idx
            print('cluster euclidean results:\n{}'.format(self.farthest_idx))
        elif self.parms.distance_type == WRONG_LABELS:
            farthest_idx = []
            farthest_idx_pro = []
            wrong_labels = self.wrong_labels
            cluster_labels = self.cluster_label_pred
            for ti, target in enumerate(self.targets):
                farthest = np.zeros(self.n_classes)
                for i in range(self.n_classes):
                    if i == cluster_labels[target]: # delete the cluster of target
                        continue
                    sl = self.sur_labels[cluster_labels == i]
                    farthest[i] = (sl == wrong_labels[ti]).mean()
                df = pd.DataFrame(farthest).sort_values(0, ascending=False)
                farthest_idx.append(np.array(df.index))
                farthest_idx_pro.append(df.values.ravel())
                print('cluster:{}, pro:{}'.format(farthest_idx[-1], farthest_idx_pro[-1]))
            self.farthest_idx = np.array(farthest_idx)


    def visualization(self):
        # print('targets labels:{}'.format(list(self.cluster_label_pred)))
        self.tsne = TSNE()
        self.tsne.fit_transform(self.z)
        X = pd.DataFrame(self.z)
        X['labels'] = self.cluster_label_pred
        # X['labels'] = self.graph.node_label
        tsne = pd.DataFrame(self.tsne.embedding_, index=X.index)  # 转换数据格式
        for label in range(self.n_classes):
            d = tsne[X[u'labels'] == label]
            plt.plot(d[0], d[1], '.', label=str(label))
        plt.legend()
        plt.show()

    def do_cluster(self):
        if self.cluster_type == "KMeans":
            self.cluser_model = KMeans(n_clusters=self.n_classes,
                                       max_iter=self.parms.max_iter,
                                       n_init=self.parms.n_init,
                                       init="k-means++",
                                       random_state=self.parms.seed)

        self.cluser_model.fit(self.z)
        self.cluster_label_pred = self.cluser_model.labels_
        self.cluster_centroidds = self.cluser_model.cluster_centers_
        self.intertia = self.cluser_model.inertia_
        self.get_farthest_idx()

        statis = [(self.cluster_label_pred == label).sum() for label in range(self.n_classes)]
        print(statis)

    @staticmethod
    @njit(cache=True)
    def get_indirect_deleted_added_nodes(targets, indices, indptr, label_pred, farthest_idx, n_nodes, z, distance_type, topk_cluster, random, is_het):
        deleted_nodes = []
        added_nodes = []
        for target in targets:
            indirect_targets = indices[indptr[target]:indptr[target + 1]]
            indirect_deleted_nodes = get_deleted_nodes(indirect_targets, indices, indptr)
            deleted_nodes.append(indirect_deleted_nodes)

            indirect_added_nodes = get_added_nodes(indirect_targets, label_pred, farthest_idx, n_nodes, z, distance_type, topk_cluster=topk_cluster, random=random, is_het=is_het)
            added_nodes.append(indirect_added_nodes)
        return deleted_nodes, added_nodes

    @staticmethod
    @njit(cache=True)
    def get_edges(targets, deleted_nodes, added_nodes):
        sub_nodes = []
        deleted_edges = []
        added_edges = []
        deleted_const = set(np.array([-1], dtype=np.int32))
        for i, target in enumerate(targets):
            dn_set = set(deleted_nodes[i]) - deleted_const
            ad_set = set(added_nodes[i]) - deleted_const
            dns = list(dn_set - ad_set)
            ans = list(ad_set - dn_set)

            sub_nodes.append(np.array(list(dn_set | ad_set)))
            deleted_edges.append(list(zip([target] * len(dns), dns)))
            added_edges.append(list(zip([target] * len(ans), ans)))
        return sub_nodes, deleted_edges, added_edges

    @staticmethod
    @njit(cache=True)
    def get_indirect_edges(targets, deleted_nodes, added_nodes, indices, indptr):
        sub_nodes = []
        deleted_edges = []
        added_edges = []
        deleted_const = set(np.array([-1], dtype=np.int32))
        for i, target in enumerate(targets):
            tmp_deleted_edges = []
            tmp_added_edges = []
            indirect_targets = indices[indptr[target]:indptr[target + 1]]
            sub_node = set(indirect_targets)
            # count = 0
            for j, indirect_target in enumerate(indirect_targets):
                dn_set = set(deleted_nodes[i][j]) - deleted_const
                ad_set = set(added_nodes[i][j]) - deleted_const
                dns = dn_set - ad_set
                ans = ad_set - dn_set
                # count += len(dns) + len(ans)
                sub_node = sub_node | dns | ans
                tmp_deleted_edges.extend(list(zip([indirect_target] * len(dns), list(dns))))
                tmp_added_edges.extend(list(zip([indirect_target] * len(ans), list(ans))))
            deleted_edges.append(tmp_deleted_edges)
            added_edges.append(tmp_added_edges)
            # print('target: {}, iter i: {}, sub_node: {}, tmp_deleted_edges: {}, tmp_added_edges:{}, total_edges: {}'.format(
            #     target, i, len(sub_node), len(tmp_deleted_edges), len(tmp_added_edges),
            #     len(tmp_deleted_edges) + len(tmp_added_edges) == count))
            sub_nodes.append(np.array(list(sub_node)))
        return sub_nodes, deleted_edges, added_edges

    def get_candidates(self):
        if self.direct_attack:
            deleted_nodes = get_deleted_nodes(self.targets, self.indices, self.indptr)
            added_nodes = get_added_nodes(self.targets, self.cluster_label_pred, self.farthest_idx,
                                          self.n_nodes, self.z, self.parms.distance_type, topk_cluster=self.parms.topk_cluster,
                                          random=self.parms.random, is_het=self.parms.is_het)
            deleted_nodes = make_redundancy(deleted_nodes)
            added_nodes = make_redundancy(added_nodes)
            self.sub_nodes, deleted_edges, added_edges = self.get_edges(self.targets, deleted_nodes, added_nodes)
        else:
            deleted_nodes, added_nodes = self.get_indirect_deleted_added_nodes(self.targets, self.indices,
                                            self.indptr, self.cluster_label_pred, self.farthest_idx, self.n_nodes,
                                            self.z, self.parms.distance_type, topk_cluster=self.parms.topk_cluster,
                                            random=self.parms.random, is_het=self.parms.is_het)
            deleted_nodes = make_redundancy(deleted_nodes, False)
            added_nodes = make_redundancy(added_nodes, False)
            self.sub_nodes, deleted_edges, added_edges = self.get_indirect_edges(self.targets, deleted_nodes, added_nodes, self.indices, self.indptr)
        self.deleted_edges = [gf.asedge(sub_edges, shape='row_wise').T if len(sub_edges) > 0 else np.array([[], []], dtype='int64') for sub_edges in deleted_edges]
        self.added_edges = [gf.asedge(sub_edges, shape='row_wise').T if len(sub_edges) > 0 else np.array([[], []], dtype='int64') for sub_edges in added_edges]
        # print(self.sub_nodes)


@njit(cache=True)
def get_deleted_nodes(targets, indices, indptr):
    deleted_nodes = []
    for target in targets:
        nbrs = indices[indptr[target]:indptr[target + 1]]
        deleted_nodes.append(nbrs)
    return deleted_nodes


@njit(cache=True)
def get_added_nodes(targets, label_pred, farthest_idx, n_nodes, z, distance_type, extra_nums_nodes=5, topk_cluster=1, random=False, is_het=False):
    if random:
        topk_cluster = farthest_idx.shape[1]
    elif topk_cluster == 1:
        random = False
    added_nodes = []
    for i, target in enumerate(targets):
        added_node = []
        if distance_type == EUCLIDEAN:
            candidate_labels = farthest_idx[label_pred[target]][:-1][:topk_cluster]
            # candidate_labels = farthest_idx[target_label_pred][::-1][1:][:topk_cluster]
        elif distance_type == WRONG_LABELS:
            candidate_labels = farthest_idx[i][:topk_cluster]

        for i, label in enumerate(candidate_labels):
            nnodes = n_nodes[label_pred == label]
            if i > 0:
                topk = min(extra_nums_nodes, len(nnodes))
                if not random:
                    farthest = np.zeros(len(nnodes))
                    for j in range(len(nnodes)):
                        farthest[j] = ((z[target] - z[nnodes[j]])**2).sum()
                    if is_het:
                        idx_topk = np.argsort(farthest)[:topk]
                    else:
                        idx_topk = np.argsort(farthest)[-topk:]
                    nnodes = nnodes[idx_topk]
                else:
                    nnodes = np.random.choice(nnodes, topk, replace=False)
            added_node.extend(nnodes)
        added_nodes.append(np.array(added_node))
    return added_nodes


def make_redundancy(arr, direct_attack=True):
    new_arr = []
    if direct_attack:
        col = 0
        for a in arr:
            col = max(col, len(a))
        for a in arr:
            new_arr.append(list(a) + [-1] * (col - len(a)))
    else:
        row = 0
        col = 0
        for a in arr:
            row = max(row, len(a))
            for b in a:
                col = max(col, len(b))
        new_arr = []
        for a in arr:
            new_brr = []
            for b in a:
                new_brr.append(list(b) + [-1] * (col - len(b)))
            row_added = - np.ones((row - len(a), col))
            new_brr.extend(row_added)
            new_arr.append(new_brr)
    return np.array(new_arr, dtype=np.int32)