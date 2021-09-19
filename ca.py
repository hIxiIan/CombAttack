from sga import *
from graphgallery.datasets import NPZDataset
import random


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
used_seeds = seeds[:1]
us = []
atks = []
for seed in used_seeds:
    print('----------', seed)
    try:
        diff1, model1 = test(seed=seed)
        diff2, model2 = test(subgraph_type='spread_random', seed=seed, p=0.5)
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
