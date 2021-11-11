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
    def __init__(self, embed_type, targets, model, graph, sample_ratio):
        self.embed_type = embed_type
        self.targets = np.array(targets)
        self.model = model
        self.graph = graph
        self.indices = self.graph.adj_matrix.indices
        self.indptr = self.graph.adj_matrix.indptr
        self.n_nodes = np.array(range(graph.adj_matrix.shape[0]))
        self.n_classes = len(set(graph.node_label))
        self.sample_nums = int(sample_ratio * graph.adj_matrix.shape[0])
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
        if "GCN" in self.embed_type:
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
            self.z = propagation(x, self.model.cache.A)
        else:
            self.z = self.model.predict(self.n_nodes)
        print('z shape: ', self.z.shape)

    def do(self):
        self.get_predict()
        self.init_cluster()
        self.get_candidates()
        # self.visualization()

    def get_farthest_idx(self):
        farthest_idx = [-1 for _ in range(self.n_classes)]
        farthest = [0 for _ in range(self.n_classes)]
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                distance = pow(self.cluster_centroidds[i] - self.cluster_centroidds[j], 2).sum()
                if farthest[i] == 0 or distance > farthest[i]:
                    farthest_idx[i] = j
                    farthest[i] = distance
                if farthest[j] == 0 or distance > farthest[j]:
                    farthest_idx[j] = i
                    farthest[j] = distance
        self.farthest_idx = np.asarray(farthest_idx)

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

    def init_cluster(self):
        if self.cluster_type == "KMeans":
            self.cluser_model = KMeans(n_clusters=self.n_classes,
                                       max_iter=300,
                                       n_init=40,
                                       init="k-means++")

        self.cluser_model.fit(self.z)
        self.cluster_label_pred = self.cluser_model.labels_
        self.cluster_centroidds = self.cluser_model.cluster_centers_
        self.intertia = self.cluser_model.inertia_
        self.get_farthest_idx()

    @staticmethod
    @njit(cache=True)
    def get_deleted_nodes(targets, indices, indptr):
        deleted_nodes = []
        for target in targets:
            nbrs = indices[indptr[target]:indptr[target + 1]]
            deleted_nodes.append(nbrs)
        return deleted_nodes

    @staticmethod
    @njit(cache=True)
    def get_added_nodes(targets, label_pred, farthest_idx, n_nodes):
        added_nodes = []
        for target in targets:
            target_label_pred = label_pred[target]
            farthest_label = farthest_idx[target_label_pred]
            farthest_nodes = n_nodes[label_pred == farthest_label]
            added_nodes.append(farthest_nodes)
        return added_nodes

    @staticmethod
    def get_edges(targets, deleted_nodes, added_nodes):
        N = len(targets)
        sub_nodes = []
        deleted_edges = []
        added_edges = []
        # 删边集合
        for i in range(N):
            deleted_edges.append([])
            added_edges.append([])
            sub_nodes.append(np.union1d(deleted_nodes[i], added_nodes[i]))
            dns = np.setdiff1d(deleted_nodes[i], added_nodes[i])
            ans = np.setdiff1d(added_nodes[i], deleted_nodes[i])
            for nbr in dns:
                deleted_edges[-1].append([targets[i], nbr])
            for nbr in ans:
                added_edges[-1].append([targets[i], nbr])
        return sub_nodes, deleted_edges, added_edges

    def get_candidates(self):
        deleted_nodes = self.get_deleted_nodes(self.targets, self.indices, self.indptr)
        added_nodes = self.get_added_nodes(self.targets, self.cluster_label_pred, self.farthest_idx, self.n_nodes)
        self.sub_nodes, deleted_edges, added_edges = self.get_edges(self.targets, deleted_nodes, added_nodes)
        self.deleted_edges = [gf.asedge(sub_edges, shape='row_wise').T if len(sub_edges) > 0 else np.array([[], []], dtype='int64') for sub_edges in deleted_edges]
        self.added_edges = [gf.asedge(sub_edges, shape='row_wise').T if len(sub_edges) > 0 else np.array([[], []], dtype='int64') for sub_edges in added_edges]



