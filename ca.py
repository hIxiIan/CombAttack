from sga import *
from graphgallery.datasets import NPZDataset
import random


def test(target=1, subgraph_type='gcn', seed=124, prob=0.5, p=2.0, q=0.25, sample_ratio=0.3, hops=2, hop_mode=False):
    verbose = 0
    ################### Surrogate model ############################
    trainer = gg.gallery.nodeclas.SGC(seed=1000).setup_graph(graph, K=2).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=verbose,
                      epochs=100)

    ################### Attacker model ############################
    attacker = SGA(graph, seed=seed).process(trainer)
    attacker.attack(target, subgraph_type=subgraph_type, prob=prob, p=p, q=q, sample_ratio=sample_ratio, hops=hops, hop_mode=hop_mode)
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

times = 10
used_seeds = random.sample(seeds, times)
atks = []
for seed in used_seeds:
    print('----------', seed)
    try:
        effectOrigin, modelOrigin = test(seed=seed)
        effectDW, modelDW = test(subgraph_type='dw', seed=seed, p=1.0, q=1.0)
        effectN2V, modelN2V = test(subgraph_type='n2v', seed=seed, p=2.0, q=0.25)
        effectSR, modelSR = test(subgraph_type='spread_random', seed=seed, prob=0.5)
    except AssertionError as e:
        print(e)
        continue
    except PermissionError as e:
        print(e)
        continue
    cur_effect = np.array([effectOrigin, effectDW, effectN2V, effectSR]) - effectOrigin
    atks.append(cur_effect)
    if any(cur_effect > 0):
        print('##########')
        print('[ORIGIN, DW, N2V, SR]:', cur_effect)
