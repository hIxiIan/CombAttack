import graphgallery as gg
import pandas as pd
from graphgallery import functional as gf
from sklearn.cluster import KMeans
from numba import njit
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch


class Cluster:
    def __init__(self, direct_attack, embed_type, targets, model, graph, sample_ratio, parms):
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
        self.parms = parms
        self.z = None
        self.tsne = TSNE()
        self.cluster_type = "KMeans"
        self.cluser_model = None
        self.cluster_label_pred = None
        self.cluster_centroidds = None
        self.intertia = None
        self.farthest_idx = None

        self.sub_nodes = None
        self.deleted_edges = None
        self.added_edges = None

        self.targets_map = {}
        for i, target in enumerate(targets):
            self.targets_map[target] = i

        self.do()

    @torch.no_grad()
    def get_predict(self):
        if self.embed_type in ["GCN", "GCN_E"]:
            conv = self.model.model.conv[:-3]
            self.z = conv(self.model.cache.X, self.model.cache.A).cpu().numpy()
        elif "MLP" in self.embed_type:
            lin = self.model.model.lin[:-3]
            self.z = lin(self.model.cache.X).cpu().numpy()
        elif "SGC" in self.embed_type:
            lin = self.model.model.lin
            self.z = lin(self.model.cache.X).cpu().numpy()
        elif "PPNP" in self.embed_type:
            lin = self.model.model.lin[:-3]
            propagation = self.model.model.propagation
            x = lin(self.model.cache.X)
            self.z = propagation(x, self.model.cache.A).cpu().numpy()
        elif self.embed_type in ["DW", 'N2V', 'BANE']:
            self.z = self.model.get_embedding()
        else:
            self.z = self.model.predict(self.n_nodes)
        # print('z shape: ', self.z.shape)

    def do(self):
        self.get_predict()
        self.do_cluster()
        self.get_candidates()
        # self.visualization()

    def compute_distance(self, embed1, embed2):
        return pow(embed1 - embed2, 2).sum()

    def get_farthest_idx(self):
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
        print('cluster results:\n{}'.format(self.farthest_idx))

    def visualization(self):
        # print('targets labels:{}'.format(list(self.cluster_label_pred)))
        self.tsne.fit_transform(self.z)
        X = pd.DataFrame(self.z)
        X['labels'] = self.cluster_label_pred
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
    def get_indirect_deleted_added_nodes(targets, indices, indptr, label_pred, farthest_idx, n_nodes, z, topk_cluster):
        deleted_nodes = []
        added_nodes = []
        for target in targets:
            indirect_targets = indices[indptr[target]:indptr[target + 1]]
            indirect_deleted_nodes = get_deleted_nodes(indirect_targets, indices, indptr)
            deleted_nodes.append(indirect_deleted_nodes)

            indirect_added_nodes = get_added_nodes(indirect_targets, label_pred, farthest_idx, n_nodes, z, topk_cluster)
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
    # @njit(cache=True)
    def get_indirect_edges(targets, deleted_nodes, added_nodes, indices, indptr):
        sub_nodes = []
        deleted_edges = []
        added_edges = []
        deleted_const = set(np.array([-1], dtype=np.int32))
        for i, target in enumerate(targets):
            tmp_deleted_edges = []
            tmp_added_edges = []
            indirect_targets = indices[indptr[target]:indptr[target + 1]]
            sub_node = set()
            count = 0
            for j, indirect_target in enumerate(indirect_targets):
                dn_set = set(deleted_nodes[i][j]) - deleted_const
                ad_set = set(added_nodes[i][j]) - deleted_const
                dns = dn_set - ad_set
                ans = ad_set - dn_set
                count += len(dns) + len(ans)
                # print('sub_node', list(sub_node))
                sub_node = sub_node | dns | ans
                # print('iter j: {}'.format(j))
                # print('len: {}, dn_set_: {}'.format(len(dn_set_), list(dn_set_)))
                # print('len: {}, ad_set_: {}'.format(len(ad_set_), list(ad_set_)))
                # print('len: {}, dns: {}'.format(len(dns), list(dns)))
                # print('len: {}, ans: {}'.format(len(ans), list(ans)))
                tmp_deleted_edges.extend(list(zip([indirect_target] * len(dns), list(dns))))
                tmp_added_edges.extend(list(zip([indirect_target] * len(ans), list(ans))))
                deleted_edges.append(tmp_deleted_edges)
                added_edges.append(tmp_added_edges)
            print('target: {}, iter i: {}, sub_node: {}, tmp_deleted_edges: {}, tmp_added_edges:{}, total_edges: {}'.format(target, i, len(sub_node), len(tmp_deleted_edges), len(tmp_added_edges), len(tmp_deleted_edges) + len(tmp_added_edges) == count))
            # print(tmp_deleted_edges)
            # print(tmp_added_edges)
            # print()
            sub_nodes.append(np.array(list(sub_node)))
        return sub_nodes, deleted_edges, added_edges

    def get_candidates(self):
        if self.direct_attack:
            deleted_nodes = get_deleted_nodes(self.targets, self.indices, self.indptr)
            added_nodes = get_added_nodes(self.targets, self.cluster_label_pred, self.farthest_idx, self.n_nodes, self.z, topk_cluster=self.parms.topk_cluster, random=self.parms.random, is_het=self.parms.is_het)
            deleted_nodes = make_redundancy(deleted_nodes)
            added_nodes = make_redundancy(added_nodes)
            self.sub_nodes, deleted_edges, added_edges = self.get_edges(self.targets, deleted_nodes, added_nodes)
        else:
            deleted_nodes, added_nodes = self.get_indirect_deleted_added_nodes(self.targets, self.indices, self.indptr, self.cluster_label_pred, self.farthest_idx, self.n_nodes, self.z, self.parms.topk_cluster)
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
def get_added_nodes(targets, label_pred, farthest_idx, n_nodes, z, extra_nums_nodes=5, topk_cluster=1, random=False, is_het=False):
    if random:
        topk_cluster = farthest_idx.shape[1]
    elif topk_cluster == 1:
        random = False
    added_nodes = []
    for target in targets:
        added_node = []
        target_label_pred = label_pred[target]
        candidate_labels = farthest_idx[target_label_pred][:-1][:topk_cluster]
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