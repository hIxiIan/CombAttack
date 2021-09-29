from sga import SCA
from orisga import SGA
from graphgallery.datasets import NPZDataset
import random
import graphgallery as gg
import numpy as np


def get_purity(adj, labels):
    indices = adj.indices
    indptr = adj.indptr
    N = adj.shape[0]
    purity = []
    for node_i in range(N):
        node_i_label = labels[node_i]
        nbrs = indices[indptr[node_i]:indptr[node_i+1]]
        nbrs_label = labels[nbrs]
        cnt = (nbrs_label == node_i_label).mean()
        purity.append(cnt)
    return purity


def testSGA(target=1, w_label=None, seed=124):
    verbose = 0
    ################### Surrogate model ############################
    trainer = gg.gallery.nodeclas.SGC(seed=1000).setup_graph(graph, K=2).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=verbose,
                      epochs=100)

    ################### Attacker model ############################
    attacker = SGA(graph, seed=seed).process(trainer)
    attacker.attack(target, direct_attack=True, w_label=w_label)
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
    target_label = graph.node_label[target]
    print('target_label:', target_label)
    print('w_label', w_label)
    print('original_predict', original_predict)
    print('perturbed_predict', perturbed_predict)
    diff = original_predict[target_label] - perturbed_predict[target_label]
    perturbed_predict_ = perturbed_predict[np.array(range(len(perturbed_predict))) != target_label]
    diffmax = original_predict[target_label] - perturbed_predict_.max()
    return (diff, diffmax), attacker


def testSCA(target=1, w_label=None, subgraph_type='dw', seed=124, prob=0.5, p=2.0, q=0.25, sample_ratio=0.3, hops=2, hop_mode=False):
    verbose = 0
    ################### Surrogate model ############################
    trainer = gg.gallery.nodeclas.SGC(seed=1000).setup_graph(graph, K=2).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=verbose,
                      epochs=100)

    ################### Attacker model ############################
    attacker = SCA(graph, seed=seed).process(trainer)
    attacker.attack(target, direct_attack=True, w_label=w_label, subgraph_type=subgraph_type, prob=prob, p=p, q=q, sample_ratio=sample_ratio, hops=hops, hop_mode=hop_mode)
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
    target_label = graph.node_label[target]
    print('target_label:', target_label)
    print('w_label', w_label)
    print('original_predict', original_predict)
    print('perturbed_predict', perturbed_predict)
    diff = original_predict[target_label] - perturbed_predict[target_label]
    perturbed_predict_ = perturbed_predict[np.array(range(len(perturbed_predict))) != target_label]
    diffmax = original_predict[target_label] - perturbed_predict_.max()
    return (diff, diffmax), attacker


if __name__ == '__main__':

    gg.set_backend("th")

    data = NPZDataset('cora',
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")

    graph = data.graph
    splits = data.split_nodes(random_state=15)
    seeds = list(range(9999))
    adj_matrix = graph.adj_matrix
    attr_matrix = graph.node_attr
    labels = graph.node_label
    purity = get_purity(adj_matrix, labels)


    seed = 0
    target = 1234
    effect_ori_tuple, model_ori = testSGA(target=target, seed=seed)
    effect_n2v_tuple, model_n2v = testSCA(target=target, subgraph_type='ppr', seed=seed)

    print(model_ori.added_edges)
    print(model_ori.non_added_edges)
    print(model_n2v.added_edges)
    print(model_n2v.non_added_edges)
    print(effect_ori_tuple, effect_n2v_tuple)