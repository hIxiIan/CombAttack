import graphgallery as gg
from graphgallery.datasets import NPZDataset

from CombAttack.sampler import Sampler

gg.set_backend("th")
data = NPZDataset('cora',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")
graph = data.graph
adj_matrix = graph.adj_matrix
p = 1.0
q = 1.0
sampler = Sampler(adj_matrix, p, q, seed=123)
