import graphgallery as gg
import time

from graphgallery.datasets import NPZDataset

from sampler import Sampler

gg.set_backend("th")

data = NPZDataset('cora',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")

graph = data.graph
adj_matrix = graph.adj_matrix
targets = [0]
sample_nums = 500

# t1 = time.time()
# p = 1.0
# q = 1.0
# sampler = Sampler(adj_matrix, p, q, seed=123)
# dw = True
# edges, nodes = sampler.random_sample(adj_matrix, targets, sample_nums, dw)
#
# print('deepwalk cost:', (time.time() - t1) / 60)
# print(edges)
# print(nodes)


p = 2.0
q = 0.25
sampler = Sampler(adj_matrix, p, q, seed=123)
dw = True
t1 = time.time()
edges, nodes = sampler.random_sample(targets, sample_nums, dw)
print('deepwalk cost:', (time.time() - t1) / 60)
print(edges)
print(nodes)

t1 = time.time()
dw = False
edges, nodes = sampler.random_sample(targets, sample_nums, dw)
print('node2vec cost:', (time.time() - t1) / 60)
print(edges)
print(nodes)

t1 = time.time()
prob = 0.5
hops = 2
hop_mode = False
edges, nodes = sampler.spread_sample(targets, prob, hops, hop_mode)
print('spread cost:', (time.time() - t1) / 60)
print(edges, edges.shape)
print(nodes, nodes.shape)
