import numpy as np
from numba import jit
from time import time
import warnings
import numba


class NGA:
    def __init__(self, targets, logits, graph, added_mode, scale):
        self.targets = np.array(targets)
        self.logits = logits
        self.graph = graph
        self.n_nodes = np.array(range(graph.adj_matrix.shape[0]))
        self.sub_nodes = None
        self.deleted_edges = None
        self.added_edges = None
        self.deg = None
        self.cet = None
        self.added_mode = added_mode
        self.scale = scale

        self.targets_map = {}
        for i, target in enumerate(targets):
            self.targets_map[target] = i

        self.do()

    def do(self):
        start = time()
        self.get_cross_entropy_targets()
        self.get_edges()
        end = time()
        self.cluster_cost_time = end - start

    def get_cross_entropy_targets(self):
        adj_matrix = self.graph.adj_matrix
        deg = np.array(np.sum(adj_matrix, axis=0))[0]
        cet = get_cross_entropy_targets(self.targets, self.logits, adj_matrix.indices, adj_matrix.indptr, deg)
        self.cet = cet
        self.deg = deg

    def get_edges(self):
        cet = np.argsort(-self.cet)
        adj_matrix = self.graph.adj_matrix
        deleted_nodes = get_deleted_nodes(self.targets, adj_matrix.indices, adj_matrix.indptr)
        if self.added_mode == "random":
            added_nodes = get_added_nodes_random(self.targets, self.deg, self.scale, self.n_nodes)
        else:
            added_nodes = get_added_nodes(self.targets, cet, self.deg, self.scale)

        sub_nodes = []
        deleted_edges = []
        added_edges = []
        for i, target in enumerate(self.targets):
            # print('target:{}, deg:{}, scale:{}, added_nodes nums:{}'.format(target, self.deg[target], self.scale, len(added_nodes[i])))
            sub_nodes.append(np.array(list(set(deleted_nodes[i]) | set(added_nodes[i]))))
            deleted_edges.append(list(zip([target] * len(deleted_nodes[i]), deleted_nodes[i])))
            added_edges.append(list(zip([target] * len(added_nodes[i]), added_nodes[i])))
        self.sub_nodes = sub_nodes
        self.deleted_edges = [asedge(sub_edges, shape='row_wise').T if len(sub_edges) > 0 else np.array([[], []], dtype='int64') for sub_edges in deleted_edges]
        self.added_edges = [asedge(sub_edges, shape='row_wise').T if len(sub_edges) > 0 else np.array([[], []], dtype='int64') for sub_edges in added_edges]


@jit(cache=True, nopython=True)
def _cross_entropy(hi, hj):
    return - np.sum(hi * np.log(hj))


@jit(cache=True, nopython=True)
def get_cross_entropy_targets(targets, logits, indices, indptr, deg):
    N = len(logits)
    TN = len(targets)
    cet = np.zeros((TN, N), dtype=np.float32)
    for tn in range(TN):
        target = targets[tn]
        nbrs = indices[indptr[target]:indptr[target + 1]]
        for node in range(N):
            ce = _cross_entropy(logits[target], logits[node]) / deg[node]
            cet[tn, node] = ce
        cet[tn][nbrs] = 0.
    return cet


@jit(cache=True, nopython=True)
def get_deleted_nodes(targets, indices, indptr):
    deleted_nodes = []
    for target in targets:
        nbrs = indices[indptr[target]:indptr[target + 1]]
        deleted_nodes.append(nbrs)
    return deleted_nodes


@jit(cache=True, nopython=True, locals={"nums": numba.int32})
def get_added_nodes(targets, cet, deg, scale):
    added_nodes = []
    for i, target in enumerate(targets):
        nums = min(deg[target] * scale, deg.shape[0])
        added_nodes.append(cet[i][:nums])
    return added_nodes


@jit(cache=True, nopython=True, locals={"nums": numba.int32})
def get_added_nodes_random(targets, deg, scale, n_nodes):
    added_nodes = []
    for target in targets:
        nums = min(deg[target] * scale, deg.shape[0])
        cur_nnodes = np.random.choice(n_nodes, nums, replace=False)
        added_nodes.append(cur_nnodes)
    return added_nodes


def asedge(edge: np.ndarray, shape="col_wise", symmetric=False, dtype=None):
    assert shape in ["row_wise", "col_wise"], shape
    assert isinstance(edge, (np.ndarray, list, tuple)), edge
    edge = np.asarray(edge, dtype=dtype or "int64")
    assert edge.ndim == 2 and 2 in edge.shape, edge.shape
    N, M = edge.shape
    if N == M == 2 and shape == "col_wise":
        # TODO: N=M=2 is confusing, we assume that edge was 'row_wise'
        warnings.warn(f"The shape of the edge is {N}x{M}."
                      f"we assume that {edge} was 'row_wise'")
        edge = edge.T
    elif (shape == "col_wise" and N != 2) or (shape == "row_wise" and M != 2):
        edge = edge.T

    if symmetric:
        if shape == "col_wise":
            edge = np.hstack([edge, edge[[1, 0]]])
        else:
            edge = np.vstack([edge, edge[:, [1, 0]]])

    return edge