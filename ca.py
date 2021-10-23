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
    t1 = time()
    walker, spreader, pprer = None, None, None

    if "dw" in args.subgraph_type or "n2v" in args.subgraph_type:
        walker = Walker(args.subgraph_type, attacker.graph.adj_matrix, attacker.graph.node_label, args.p, args.q, attacker.softmax_logits, wl_limit=args.wl_limit)
    elif "spread" in args.subgraph_type:
        spreader = Spreader(args.subgraph_type, attacker.graph.adj_matrix, attacker.graph.node_label, args.prob, args.hops, attacker.logits)
    elif "ppr" in args.subgraph_type:
        pprer = PPRer(args.subgraph_type, attacker.graph.adj_matrix, attacker.graph.node_label, args.alpha, attacker.softmax_logits, args.wl_limit, args.eps)

    print('init_sampler end..., cost:{} min'.format((time() - t1) / 60))
    return walker, spreader, pprer


def testACC(gcn_model, attacker, args, us=True, verbose=True, verbose_us=False):
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
                attacker.attack(target, walker=walker, spreader=spreader, pprer=pprer, with_w_label=args.with_w_label, verbose_us=verbose_us, direct_attack=args.direct_attack,
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
        his = trainer.fit(args.splits.train_nodes,
                          args.splits.val_nodes,
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

        if args.subgraph_type != "sga":
            if attacker.with_w_label:
                print('iter: {}, attack target node {}, add all of wrong label nodes to subgraph'.format(i, target))
            else:
                if attacker._walk_length <= 10:
                    print('iter: {}, attack target node {}, subgraph length <= 10'.format(i, target))
        if verbose:
            print('###################')
            print('iter: {}, attack target node {}, get subgraph cost:{}, attack cost: {} min'.format(i, target, attacker._subgraph_time, (end_i - start_i) / 60))
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
    cost = (end - start) / 60
    print('subgraph:{}, p:{}, q:{}, alpha:{}'.format(args.subgraph_type, args.p, args.q, args.alpha))
    print('testACC end, cost time: {} min'.format(cost))

    return acc, wlacc, cost


def run(subgraph_type, cmd=None, with_w_label=False, sample_ratio=0.05, p=2.0, q=0.25, alpha=0.25, us=True, verbose=True):
    data = NPZDataset(cmd.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")

    graph = data.graph
    splits = data.split_nodes(random_state=15)
    targets = random.sample(list(splits.test_nodes), 50)
    args = ARGS(cmd=cmd, targets=targets, splits=splits)
    random.seed(args.seed)

    args.subgraph_type = subgraph_type
    args.with_w_label = with_w_label
    args.p = p
    args.q = q
    args.alpha = alpha
    surrogate_model = gg.gallery.nodeclas.SGC(device=args.device, seed=1000).setup_graph(graph, K=2).build()
    his = surrogate_model.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=args.verbose,
                      epochs=100)
    args.surrogate_model = surrogate_model
    # Before attack
    gcn_model = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(graph).build()
    his = gcn_model.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=args.verbose,
                      epochs=100)


    # attacker
    if us:
        attacker = SCA(graph, device=args.device, seed=args.seed).process(surrogate_model)
    else:
        attacker = SGA(graph, device=args.device, seed=args.seed).process(surrogate_model)
    acc, wlacc, cost = testACC(gcn_model, attacker, args, us=us, verbose=verbose)
    return acc, wlacc, cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"], help="code environment")

    parser.add_argument("-st", "--subgraph_type", default="dw_wl", type=str, help="sample method")
    parser.add_argument("-sr", "--sample_ratio", default=0.05, type=float, help="ratio of sampled nodes")
    parser.add_argument("-in_da", "--indirect_attack", action="store_true", help="indirect attack")

    parser.add_argument("--dataset", default="cora", type=str, help="dataset")
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    parser.add_argument("-p", default=7.0, type=float)
    parser.add_argument("-q", default=0.25, type=float)
    parser.add_argument("-a", "--alpha", default=0.25, type=float)

    cmd = parser.parse_args()
    # cmd.dataset = "cora_full"
    random.seed(cmd.seed)
    gg.set_backend("th")
    data = NPZDataset(cmd.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")

    graph = data.graph
    splits = data.split_nodes(random_state=15)
    targets = random.sample(list(splits.test_nodes), 50)
    args = ARGS(cmd=cmd, targets=targets, splits=splits)
    args.subgraph_type = "n2v_wl"
    args.seed = 2012
    # args.p = 7.0
    # args.q = 0.25
    # args.alpha = 0.01
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
        attacker = SCA(graph, device=args.device, seed=args.seed).process(surrogate_model)
    else:
        attacker = SGA(graph, device=args.device, seed=args.seed).process(surrogate_model)
    acc, wlacc, cost = testACC(gcn_model, attacker, args, us=args.us, verbose_us=False)

    # attacker = SGA(graph, seed=seed).process(surrogate_model)
    # testACC(gcn_model, attacker, args, us=False)




