import random
import graphgallery as gg
import numpy as np
import networkx as nx
import pandas as pd
import argparse

from graphgallery.datasets import NPZDataset
from sga import SCA
from orisga import SGA
from time import time
from args import ARGS
from spreader import Spreader
from walker import Walker
from ppr import PPRer


def testSGA(target=1, w_label=None, device="cpu", seed=124):
    verbose = 0
    ################### Surrogate model ############################
    trainer = gg.gallery.nodeclas.SGC(device=device, seed=1000).setup_graph(graph, K=2).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=verbose,
                      epochs=100)

    ################### Attacker model ############################
    attacker = SGA(graph, device=device, seed=seed).process(trainer)
    attacker.attack(target, direct_attack=True, w_label=w_label)
    attacker.name = "ori"
    ################### Victim model ############################
    # Before attack
    trainer = gg.gallery.nodeclas.GCN(device=device, seed=seed).setup_graph(graph).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=verbose,
                      epochs=100)
    original_predict = trainer.predict(target, transform="softmax")

    # After attack
    trainer = gg.gallery.nodeclas.GCN(device=device, seed=seed).setup_graph(attacker.g).build()
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
    attacker.original_predict = original_predict
    attacker.perturbed_predict = perturbed_predict
    attacker.before = original_predict.max() - original_predict[target_label]
    attacker.before_label = original_predict.argmax()
    attacker.after = perturbed_predict.max() - perturbed_predict[target_label]
    attacker.after_label = perturbed_predict.argmax()
    # diff = original_predict[target_label] - perturbed_predict[target_label]
    # attacker.max_label = perturbed_predict.argmax()
    # diffmax = original_predict[target_label] - perturbed_predict.max()
    return attacker


def testSCA(target=1, w_label=None, subgraph_type='dw', device="cpu", seed=124, prob=0.5, p=2.0, q=0.25, sample_ratio=0.3, hops=2, keep_hops=False):
    verbose = 0
    ################### Surrogate model ############################
    trainer = gg.gallery.nodeclas.SGC(device=device, seed=1000).setup_graph(graph, K=2).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=verbose,
                      epochs=100)

    ################### Attacker model ############################
    attacker = SCA(graph, device=device, seed=seed).process(trainer)
    attacker.name = subgraph_type
    attacker.attack(target, direct_attack=True, w_label=w_label, subgraph_type=subgraph_type, prob=prob, p=p, q=q, sample_ratio=sample_ratio, hops=hops, keep_hops=keep_hops)
    ################### Victim model ############################
    # Before attack
    trainer = gg.gallery.nodeclas.GCN(device=device, seed=seed).setup_graph(graph).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=verbose,
                      epochs=100)
    original_predict = trainer.predict(target, transform="softmax")

    # After attack
    trainer = gg.gallery.nodeclas.GCN(device=device, seed=seed).setup_graph(attacker.g).build()
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
    attacker.original_predict = original_predict
    attacker.perturbed_predict = perturbed_predict
    attacker.before = original_predict.max() - original_predict[target_label]
    attacker.before_label = original_predict.argmax()
    attacker.after = perturbed_predict.max() - perturbed_predict[target_label]
    attacker.after_label = perturbed_predict.argmax()
    # diff = original_predict[target_label] - perturbed_predict[target_label]
    # attacker.max_label = perturbed_predict.argmax()
    # diffmax = original_predict[target_label] - perturbed_predict.max()
    return attacker


def print_(model, labels):
    print('###################')
    print('model_name:', model.name)
    print('true_label:', model.true_label)
    print('wrong_label:', model.wrong_label)
    print('model_added_edges', model.added_edges)
    added_edges_node_labels = [labels[edge[1]] for edge in model.added_edges]
    print('model_added_edges_labels', added_edges_node_labels)
    print('model_non_added_edges', model.non_added_edges)
    non_added_edges_node_labels = [labels[edge[1]] for edge in model.non_added_edges]
    print('model_non_added_edges_labels', non_added_edges_node_labels)
    print('original_predict', model.original_predict)
    print('perturbed_predict', model.perturbed_predict)
    print('before_atk_effect, classify to label {}, max_prob - target_prob: {}'.format(model.before_label, model.before))
    print('after_atk_effect, classify to label {}, max_prob - target_prob: {}'.format(model.after_label, model.after))
    print('')


def print_sp(model, sp):
    print('###################')
    print('model_name:', model.name)
    for edge in model.added_edges:
        print('source {} to target {}: {}'.format(model.target, edge[1], sp[edge[1]]))
    print('')


def init_sampler(attacker, args):
    walker = None
    spreader = None
    pprer = None

    if args.subgraph_type[:2] == "dw":
        walker = Walker(attacker.graph.adj_matrix, attacker.graph.node_label, args.p, args.q, attacker.softmax_logits, level_limit=args.level_limit, wl_limit=args.wl_limit)
    elif args.subgraph_type[:3] == "n2v":
        if args.subgraph_type == "n2v_purity":
            args.is_purity_matrix = True
        elif args.subgraph_type == "n2v_wl":
            args.is_wl_matrix = True
        elif args.subgraph_type == "n2v_ce":
            args.is_ce_matrix = True
        walker = Walker(attacker.graph.adj_matrix, attacker.graph.node_label, args.p, args.q, attacker.softmax_logits, args.is_purity_matrix, args.is_wl_matrix, args.is_ce_matrix, wl_limit=args.wl_limit)
    elif args.subgraph_type[:6] == "spread":
        if args.subgraph_type == "spread_random_wl_keep_hops":
            args.keep_hops = True
        if args.subgraph_type == "spread_random_ce_keep_hops":
            args.keep_hops = True
        if args.subgraph_type == "spread_ce_keep_hops":
            args.keep_hops = True
        spreader = Spreader(attacker.graph.adj_matrix, attacker.graph.node_label, args.prob, args.hops, args.keep_hops, attacker.logits)
    elif args.subgraph_type[:3] == "ppr":
        pprer = PPRer(attacker.graph.adj_matrix, attacker.graph.node_label, args.alpha, attacker.softmax_logits, args.eps)

    print('init_sampler end...')
    return walker, spreader, pprer


def testACC(gcn_model, attacker, args, us=True, verbose=True):
    if us:
        walker, spreader, pprer = init_sampler(attacker, args)
    start = time()
    res = np.zeros(len(args.targets)).astype('bool')
    res2 = np.zeros(len(args.targets)).astype('bool')
    for i, target in enumerate(args.targets):
        original_predict = gcn_model.predict(target, transform="softmax")
        start_i = time()
        attacker = attacker.reset()
        try:
            if us:
                attacker.attack(target, walker=walker, spreader=spreader, pprer=pprer, with_w_label=args.with_w_label, verbose_us=False, direct_attack=args.direct_attack,
                                subgraph_type=args.subgraph_type, sample_ratio=args.sample_ratio)
            else:
                attacker.attack(target, verbose_us=False, direct_attack=args.direct_attack)
        except AssertionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        except PermissionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))

        end_i = time()
        # After attack
        trainer = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(attacker.g).build()
        his = trainer.fit(splits.train_nodes,
                          splits.val_nodes,
                          verbose=args.verbose,
                          epochs=100)
        perturbed_predict = trainer.predict(target, transform="softmax")

        true_label = attacker.graph.node_label[target]
        perturbed_label = perturbed_predict.argmax()
        wrong_label = int(attacker.wrong_label[0])
        max_label_prob_sub_perturbed_label_prob = perturbed_predict.max() - perturbed_predict[true_label]
        if perturbed_label != true_label:
            res[i] = True
        if perturbed_label == wrong_label:
            res2[i] = True
        if verbose:
            print('###################')
            print('iter: {}, attack target node {}, cost: {} min'.format(i, target, (end_i - start_i) / 60))
            print('hop_ratio:{}, hop_length:{}, walk_length:{}'.format(attacker._hop_ratio, attacker._hop_length, attacker._walk_length))
            print('wrong_ratio:{}, wrong_length:{}'.format(attacker._wrong_ratio, attacker._wrong_length))
            print('added_edges.shape:{}, added_edges:{}'.format(len(attacker.added_edges), attacker.added_edges))
            print('deleted_edges.shape:{}, deleted_edges:{}'.format(len(attacker.non_added_edges), attacker.non_added_edges))
            print('original_predict, true_label: {}, true_label_prob: {}'.format(true_label, original_predict[true_label]))
            print('perturbed_predict, true_label_prob:{}'.format(perturbed_predict[true_label]))
            print('perturbed_predict, perturbed_label_prob:{}'.format(perturbed_predict[perturbed_label]))
            print('perturbed_predict, wrong_label_prob:{}'.format(perturbed_predict[wrong_label]))
            print('target node {}, true_label: {}, mislead to label: {}, max_label_prob_sub_perturbed_label_prob: {}'.format(
                target, true_label, perturbed_label, max_label_prob_sub_perturbed_label_prob))
            print('current acc: {}'.format(res[:i + 1].sum() / (i + 1)))
            print('current wrong label acc: {}'.format(res2[:i + 1].sum() / (i + 1)))
            print('\n\n\n')
    acc = res.sum() / len(res)
    wlacc = res2.sum() / len(res2)
    print('acc: {}'.format(acc))
    print('wrong label acc: {}'.format(wlacc))
    end = time()
    if verbose:
        print('testACC end, cost time: {} min'.format((end - start) / 60))
    return acc, wlacc


def run(subgraph_type, with_w_label=False, sample_ratio=0.05, p=2.0, q=0.25, alpha=0.25, level_limit=0, us=True):
    data = NPZDataset(cmd.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")

    graph = data.graph
    splits = data.split_nodes(random_state=15)
    targets = random.sample(list(splits.test_nodes), 50)
    args = ARGS(cmd=cmd)
    args.targets = targets
    random.seed(args.seed)

    args.subgraph_type = subgraph_type
    args.with_w_label = with_w_label
    args.p = p
    args.q = q
    args.alpha = alpha
    args.level_limit = level_limit
    surrogate_model = gg.gallery.nodeclas.SGC(seed=1000).setup_graph(graph, K=2).build()
    his = surrogate_model.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=args.verbose,
                      epochs=100)
    args.surrogate_model = surrogate_model
    # Before attack
    gcn_model = gg.gallery.nodeclas.GCN(seed=args.seed).setup_graph(graph).build()
    his = gcn_model.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=args.verbose,
                      epochs=100)


    # attacker
    if us:
        attacker = SCA(graph, seed=args.seed).process(surrogate_model)
    else:
        attacker = SGA(graph, seed=args.seed).process(surrogate_model)
    acc, wlacc = testACC(gcn_model, attacker, args, us=us)
    return acc, wlacc

# if __name__ == '__main__':
#     gg.set_backend("th")
#     data = NPZDataset('cora',
#                       root="~/GraphData/datasets/",
#                       verbose=False,
#                       transform="standardize")
#
#     graph = data.graph
#     splits = data.split_nodes(random_state=15)
#     seeds = list(range(9999))
#     adj_matrix = graph.adj_matrix
#     attr_matrix = graph.node_attr
#     labels = graph.node_label
#     g = nx.DiGraph(adj_matrix)
#
#     seed = 0
#     target = 1235
#     model_ori = testSGA(target=target, seed=seed)
#     model_n2v = testSCA(target=target, subgraph_type="dw", seed=seed, sample_ratio=0.1)
#     print_(model_ori, labels)
#     print_(model_n2v, labels)
#
#     sp = nx.shortest_path(g, source=target)
#     print_sp(model_ori, sp)
#     print_sp(model_n2v, sp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "gpu"], help="code environment")

    parser.add_argument("-st", "--subgraph_type", default="dw_wl", type=str, help="sample method")
    parser.add_argument("-sr", "--sample_ratio", default=0.05, type=float, help="ratio of sampled nodes")
    parser.add_argument("-in_da", "--indirect_attack", action="store_true", help="indirect attack")

    parser.add_argument("--dataset", default="cora", type=str, help="dataset")
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    cmd = parser.parse_args()
    random.seed(cmd.seed)
    gg.set_backend("th")
    data = NPZDataset(cmd.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")

    graph = data.graph
    splits = data.split_nodes(random_state=15)
    targets = random.sample(list(splits.test_nodes), 50)
    args = ARGS(cmd=cmd)
    args.targets = targets
    # args.subgraph_type = "dw_wl"

    surrogate_model = gg.gallery.nodeclas.SGC(device=args.device, seed=1000).setup_graph(graph, K=2).build()
    his = surrogate_model.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=args.verbose,
                      epochs=100)

    # Before attack
    gcn_model = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(graph).build()
    his = gcn_model.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=args.verbose,
                      epochs=100)

    # attacker
    if args.us:
        attacker = SCA(graph, seed=args.seed).process(surrogate_model)
    else:
        attacker = SGA(graph, seed=args.seed).process(surrogate_model)
    acc, wlacc = testACC(gcn_model, attacker, args, us=args.us)

    # attacker = SGA(graph, seed=seed).process(surrogate_model)
    # testACC(gcn_model, attacker, args, us=False)




