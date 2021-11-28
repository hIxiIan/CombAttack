import random
import graphgallery as gg
import numpy as np
import pandas as pd
import gc
import argparse
import inspect

from graphgallery.datasets import NPZDataset
from sga import SCA, SCAPD
from orisga import SGA, SGAPD
from time import time
from args import ARGS
from spreader import Spreader
from walker import Walker
from ppr import PPRer
from pd import get_lgb_model
from cluster import Cluster
from gpu_mem_track import MemTracker

DATASET_BLOCKCHAIN = ['blockchain30000', 'blockchain40000', 'blockchain50000']


def get_embed_model(args, graph):
    model = None
    if args.embed_type == "MLP":
        model = gg.gallery.nodeclas.MLP(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif args.embed_type == "GraphMLP":
        tau = 2.0
        alpha = 10.0
        if args.dataset == "cora":
            tau = 1
            alpha = 10.0
        elif args.dataset == "citeseer":
            tau = 0.5
            alpha = 1.0
        elif args.dataset == "pubmed":
            tau = 1
            alpha = 100
        model = gg.gallery.nodeclas.GraphMLP(device=args.device, seed=args.seed).setup_graph(graph).build(tau=tau, alpha=alpha)
    elif args.embed_type == "SGC":
        model = gg.gallery.nodeclas.SGC(device=args.device, seed=args.seed).setup_graph(graph, K=2).build()
    elif args.embed_type == "GCN":
        model = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif args.embed_type == "PPNP":
        model = gg.gallery.nodeclas.PPNP(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif args.embed_type == "APPNP":
        model = gg.gallery.nodeclas.APPNP(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif args.embed_type == "SimPGCN":
        model = gg.gallery.nodeclas.SimPGCN(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif args.embed_type == "GraphSAGE":
        model = gg.gallery.nodeclas.GraphSAGE(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif args.embed_type == "GAT":
        model = gg.gallery.nodeclas.GAT(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif args.embed_type == "FastGCN":
        model = gg.gallery.nodeclas.FastGCN(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif args.embed_type == "ClusterGCN":
        model = gg.gallery.nodeclas.ClusterGCN(device=args.device, seed=args.seed).setup_graph(graph, num_clusters=10).build()

    # embed_nums
    elif args.embed_type == "GCN_E":
        model = gg.gallery.nodeclas.GCN_E(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif args.embed_type == "DW":
        model = gg.gallery.embedding.DeepWalk()
    elif args.embed_type == "N2V":
        model = gg.gallery.embedding.Node2Vec(p=args.p, q=args.q)
    elif args.embed_type == "BANE":
        model = gg.gallery.embedding.BANE()

    if args.embed_type not in ["DW", 'N2V', 'BANE']:
        model.fit(args.splits.train_nodes, args.splits.val_nodes, verbose=0, epochs=100)
        results = model.evaluate(args.splits.test_nodes, verbose=0)
        print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
    else:
        if args.embed_type in ["DW", "N2V"]:
            model.fit(graph.adj_matrix)
        elif args.embed_type == "BANE":
            model.fit(graph.adj_matrix, graph.node_attr)
        results = model.evaluate_nodeclas(graph.node_label,
                                             args.splits.train_nodes,
                                             args.splits.test_nodes)
        print('Test accuracy:{}'.format(results.accuracy))
    model.embed_acc = results.accuracy
    return model


def init_sampler(attacker, args):
    if not args.us:
        return None

    sampler = None
    t1 = time()
    if not args.cluster:
        if "dw" in args.subgraph_type or "n2v" in args.subgraph_type:
            sampler = Walker(args.targets, args.sample_ratio, args.subgraph_type, attacker.graph.adj_matrix, attacker.graph.node_label, args.p, args.q, attacker.logits, attacker.softmax_logits, wl_limit=args.wl_limit)
            sampler.random_walk()
        elif "spread" in args.subgraph_type:
            sampler = Spreader(args.targets, args.sample_ratio, args.subgraph_type, attacker.graph.adj_matrix, attacker.graph.node_label, args.prob, args.hops, attacker.logits, attacker.softmax_logits)
            sampler.spread_walk()
        elif "ppr" in args.subgraph_type:
            sampler = PPRer(args.targets, args.sample_ratio, args.subgraph_type, attacker.graph.adj_matrix, attacker.graph.node_label, args.alpha, attacker.logits, attacker.softmax_logits, args.wl_limit, args.eps)
            sampler.ppr_walk()
        sampler.type_ = args.subgraph_type
        sampler.embed_acc = 0
        sampler.cluster_cost_time = 0
        print('subgraph_type:{}, sample process end..., cost:{} min'.format(args.subgraph_type, (time() - t1) / 60))
    else:
        model = get_embed_model(args, attacker.graph)
        sampler = Cluster(args.direct_attack, args.embed_type, args.targets, model, attacker.graph, args.sample_ratio, args.cluster_parms)
        sampler.type_ = args.subgraph_type
        print('embed_type:{}, sample process end..., cost:{} min'.format(args.embed_type, (time() - t1) / 60))
        sampler.embed_acc = model.embed_acc
    return sampler


def testACC(attacked_model, attacker, args, verbose=True, verbose_us=False):
    sampler = init_sampler(attacker, args)
    start = time()
    eva_res = np.zeros(len(args.targets)).astype('bool')
    eva_res_wl = np.zeros(len(args.targets)).astype('bool')
    poi_res = np.zeros(len(args.targets)).astype('bool')
    poi_res_wl = np.zeros(len(args.targets)).astype('bool')
    original_predict = attacked_model.predict(args.targets, transform="softmax")
    cost_targets = 0.
    for i, target in enumerate(args.targets):
        attacker = attacker.reset()
        start_i = time()
        try:
            if args.us:
                attacker.attack(target, sampler=sampler, verbose_us=verbose_us, direct_attack=args.direct_attack)
            else:
                attacker.attack(target, verbose_us=False, direct_attack=args.direct_attack)
        except AssertionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        except PermissionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        end_i = time()
        cost_targets = end_i - start_i
        # After attack
        true_label = original_predict[i].argmax()
        wrong_label = int(attacker.wrong_label[0])
        # evasion
        attacked_model.setup_graph(attacker.g)
        if args.atk_model_type == "SimPGCN":
            attacked_model.model.cache['adj_knn'] = attacked_model.cache['knn_graph']
        eva_predict = attacked_model.predict(target, transform="softmax")
        eva_perturbed_label = eva_predict.argmax()
        eva_max_label_prob_sub_perturbed_label_prob = eva_predict.max() - eva_predict[true_label]
        if eva_perturbed_label != true_label:
            eva_res[i] = True
            if eva_perturbed_label == wrong_label:
                eva_res_wl[i] = True

        # poisoning
        if args.atk_model_type == "GCN":
            trainer = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(attacker.g).build()
        elif args.atk_model_type == "GCN_Jaccard":
            trainer = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(attacker.g, graph_transform="jaccard_detection").build()
        elif args.atk_model_type == "SimPGCN":
            trainer = gg.gallery.nodeclas.SimPGCN(device=args.device, seed=args.seed).setup_graph(attacker.g).build()
        elif args.atk_model_type == "RobustGCN":
            trainer = gg.gallery.nodeclas.RobustGCN(device=args.device, seed=args.seed).setup_graph(attacker.g).build()

        trainer.fit(args.splits.train_nodes,
                          args.splits.val_nodes,
                          verbose=args.verbose,
                          epochs=100)
        perturbed_predict = trainer.predict(target, transform="softmax")
        perturbed_label = perturbed_predict.argmax()
        max_label_prob_sub_perturbed_label_prob = perturbed_predict.max() - perturbed_predict[true_label]
        if perturbed_label != true_label:
            poi_res[i] = True
            if perturbed_label == wrong_label:
                poi_res_wl[i] = True

        if verbose:
            print('###################')
            print('iter: {}, attack target node {}, get subgraph cost:{}, attack cost: {} min'.format(i, target, 0, (end_i - start_i) / 60))
            print('deleted_edges.shape:{}, added_edges.shape:{}, total_nodes:{}'.format(attacker._hop_ratio, attacker._hop_length, attacker._walk_length))
            print('wrong_ratio:{}, wrong_length:{}'.format(attacker._wrong_ratio, attacker._wrong_length))
            print('added_edges.shape:{}, added_edges:{}'.format(len(attacker.added_edges), attacker.added_edges))
            print('deleted_edges.shape:{}, deleted_edges:{}'.format(len(attacker.non_added_edges), attacker.non_added_edges))
            print('original_predict, true_label: {}, true_label_prob: {}'.format(true_label, original_predict[i][true_label]))

            print('#####evasion')
            print('eva_predict, true_label: {}, true_label_prob: {}'.format(true_label, eva_predict[true_label]))
            print('eva_predict, perturbed_label: {}, perturbed_label_prob:{}'.format(eva_perturbed_label, eva_predict[eva_perturbed_label]))
            print('eva_predict, wrong_label: {}, wrong_label_prob:{}'.format(wrong_label, eva_predict[wrong_label]))
            print('target node {}, mislead to label: {}, max_label_prob_sub_perturbed_label_prob: {}'.format(
                target, eva_perturbed_label, eva_max_label_prob_sub_perturbed_label_prob))
            print('current eva_asr: {}, eva_asr_wl: {}'.format(eva_res[:i + 1].sum() / (i + 1), eva_res_wl[:i + 1].sum() / (i + 1)))

            print('#####poisoning')
            print('perturbed_predict, true_label: {}, true_label_prob: {}'.format(true_label, perturbed_predict[true_label]))
            print('perturbed_predict, perturbed_label: {}, perturbed_label_prob:{}'.format(perturbed_label, perturbed_predict[perturbed_label]))
            print('perturbed_predict, wrong_label: {}, wrong_label_prob:{}'.format(wrong_label, perturbed_predict[wrong_label]))
            print('target node {}, mislead to label: {}, max_label_prob_sub_perturbed_label_prob: {}'.format(
                target, perturbed_label, max_label_prob_sub_perturbed_label_prob))
            print('current poi_asr: {}, poi_asr_wl: {}'.format(poi_res[:i + 1].sum() / (i + 1), poi_res_wl[:i + 1].sum() / (i + 1)))
            print('\n\n\n')
    eva_asr = eva_res.sum() / len(eva_res)
    eva_asr_wl = eva_res_wl.sum() / len(eva_res_wl)
    poi_asr = poi_res.sum() / len(poi_res)
    poi_asr_wl = poi_res_wl.sum() / len(poi_res_wl)
    print('eva_asr: {}, eva_asr_wl: {}'.format(eva_asr, eva_asr_wl))
    print('poi_asr: {}, poi_asr_wl: {}'.format(poi_asr, poi_asr_wl))
    end = time()
    cost = (end - start) / 60
    embed_acc = sampler.embed_acc if sampler is not None else 0
    cluster_cost_time = sampler.cluster_cost_time if sampler is not None else 0
    if args.subgraph_type != "cluster":
        print('subgraph:{}, p:{}, q:{}, alpha:{}'.format(args.subgraph_type, args.p, args.q, args.alpha))
    else:
        print('embed_type:{}, embed_acc:{}'.format(args.embed_type, embed_acc))
    print('testACC end, cost time: {} min'.format(cost))
    return [eva_asr, eva_asr_wl, poi_asr, poi_asr_wl, cost, embed_acc, cost_targets / len(args.targets), cluster_cost_time]


def get_pd(attacked_model, args):
    embedded_features = attacked_model.predict(args.train_nodes)
    original_features = args.node_attr
    true_labels = args.node_label
    combined_features_labels = np.hstack((original_features, embedded_features, true_labels.reshape(-1,1)))

    columns_name = ['f%02d' % i for i in range(combined_features_labels.shape[1] - 1)] + ['label']
    df_combined = pd.DataFrame(data=combined_features_labels, columns=columns_name, dtype=float)
    df_combined['label'] = df_combined['label'].astype(int)

    y_cols_name = ['label']
    x_cols_name = [x for x in df_combined.columns if x not in y_cols_name]

    x = df_combined[x_cols_name]
    y = df_combined[y_cols_name]
    test_res, all_predict, lgb_model = get_lgb_model(x, y, args.seed)
    # print(test_res)
    return all_predict, lgb_model


def get_train_x(attacked_model, args, target):
    embedded_features = attacked_model.predict(args.train_nodes)
    original_features = args.node_attr
    combined_features_labels = np.hstack((original_features, embedded_features))

    columns_name = ['f%02d' % i for i in range(combined_features_labels.shape[1])]
    df_combined = pd.DataFrame(data=combined_features_labels, columns=columns_name, dtype=float)

    x_cols_name = [x for x in df_combined.columns if x != 'label']
    train_x = df_combined[x_cols_name].iloc[[target]]
    return train_x


def testBlockACC(attacked_model, attacker, args, verbose=True, verbose_us=False):
    original_predict, lgb_model = get_pd(attacked_model, args)
    if args.is_phi:
        surrogate_phishing_targets = np.where(original_predict == 1)[0]
        true_phishing_targets = np.where(args.node_label == 1)[0]
        args.targets = np.intersect1d(surrogate_phishing_targets, true_phishing_targets)
        print('attack {} phishing nodes, total true phishing nodes:{}, total surrogate_phishing_nodes:{}'.format(
            len(args.targets), len(true_phishing_targets), len(surrogate_phishing_targets)))
    else:
        print('attack phishing or non-phishing nodes')

    sampler = init_sampler(attacker, args)
    eva_res = np.zeros(len(args.targets)).astype('bool')
    poi_res = np.zeros(len(args.targets)).astype('bool')
    start = time()
    cost_targets = 0.0
    for i, target in enumerate(args.targets):
        attacker = attacker.reset()
        start_i = time()
        try:
            if args.us:
                attacker.attack(target, sampler=sampler, verbose_us=verbose_us, direct_attack=args.direct_attack, blockchain=args.blockchain)
            else:
                attacker.attack(target, verbose_us=False, direct_attack=args.direct_attack, blockchain=args.blockchain)
        except AssertionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        except PermissionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        end_i = time()
        cost_targets = end_i - start_i
        # After attack
        true_label = original_predict[target]
        # evasion
        attacked_model.setup_graph(attacker.g)
        if args.atk_model_type == "SimPGCN":
            attacked_model.model.cache['adj_knn'] = attacked_model.cache['knn_graph']
        eva_perturbed_label = np.argmax(lgb_model.predict(get_train_x(attacked_model, args, target), num_iteration=lgb_model.best_iteration), axis=1)
        if eva_perturbed_label != true_label:
            eva_res[i] = True
        # poisoning
        trainer = gg.gallery.nodeclas.SGCPD(device=args.device, seed=args.seed).setup_graph(attacker.g, K=1).build()
        trainer.fit(args.train_nodes,
                    None,
                    verbose=args.verbose,
                    epochs=6)
        perturbed_predict, _ = get_pd(trainer, args)
        perturbed_label = perturbed_predict[target]
        if perturbed_label != true_label:
            poi_res[i] = True
        if verbose:
            print('###################')
            print('iter: {}, attack target node {}, get subgraph cost:{}, attack cost: {} min'.format(i, target, 0, (end_i - start_i) / 60))
            print('deleted_edges.shape:{}, added_edges.shape:{}, total_nodes:{}'.format(attacker._hop_ratio, attacker._hop_length, attacker._walk_length))
            print('wrong_ratio:{}, wrong_length:{}'.format(attacker._wrong_ratio, attacker._wrong_length))
            print('added_edges.shape:{}, added_edges:{}'.format(len(attacker.added_edges), attacker.added_edges))
            print('deleted_edges.shape:{}, deleted_edges:{}'.format(len(attacker.non_added_edges), attacker.non_added_edges))
            print('original_predict, true_label: {}'.format(true_label))
            print('#####evasion')
            print('target node {}, mislead to label: {}, current eva_asr: {}'.format(target, eva_perturbed_label, eva_res[:i + 1].sum() / (i + 1)))
            print('#####poisoning')
            print('target node {}, mislead to label: {}, current poi_asr: {}'.format(target, perturbed_label, poi_res[:i + 1].sum() / (i + 1)))
            print('\n\n\n')
    eva_asr = eva_res.sum() / len(eva_res)
    poi_asr = poi_res.sum() / len(poi_res)
    print('eva_asr:{}, poi_asr: {}'.format(eva_asr, poi_asr))
    end = time()
    cost = (end - start) / 60
    embed_acc = sampler.embed_acc if sampler is not None else 0
    cluster_cost_time = sampler.cluster_cost_time if sampler is not None else 0
    if args.subgraph_type != "cluster":
        print('subgraph:{}, p:{}, q:{}, alpha:{}'.format(args.subgraph_type, args.p, args.q, args.alpha))
    else:
        print('embed_type:{}, embed_acc:{}'.format(args.embed_type, embed_acc))
    print('testBlockACC end, cost time: {} min'.format(cost))
    print('embed_acc:{}'.format(embed_acc))

    return [eva_asr, 0, poi_asr, 0, cost, embed_acc, cost_targets / len(args.targets), cluster_cost_time]


def get_attack_model(args, graph):
    if args.dataset not in DATASET_BLOCKCHAIN:
        args.blockchain = False
        surrogate_model = gg.gallery.nodeclas.SGC(device=args.device, seed=1000).setup_graph(graph, K=2).build()
        surrogate_model.fit(args.splits.train_nodes,
                                  args.splits.val_nodes,
                                  verbose=args.verbose,
                                  epochs=100)

        # Before attack
        if args.atk_model_type == "GCN":
            attacked_model = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(graph).build()
        elif args.atk_model_type == "GCN_Jaccard":
            attacked_model = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(graph, graph_transform="jaccard_detection").build()
        elif args.atk_model_type == "SimPGCN":
            attacked_model = gg.gallery.nodeclas.SimPGCN(device=args.device, seed=args.seed).setup_graph(graph).build()
        elif args.atk_model_type == "RobustGCN":
            attacked_model = gg.gallery.nodeclas.RobustGCN(device=args.device, seed=args.seed).setup_graph(graph).build()


        # dgl backend
        elif args.atk_model_type == "MixHop":
            attacked_model = gg.gallery.nodeclas.MixHop(device=args.device, seed=args.seed).setup_graph(graph).build()

        attacked_model.fit(args.splits.train_nodes,
                           args.splits.val_nodes,
                           verbose=args.verbose,
                           epochs=100)
        if args.us:
            attacker = SCA(graph, device=args.device, seed=args.seed).process(surrogate_model)
        else:
            attacker = SGA(graph, device=args.device, seed=args.seed).process(surrogate_model)
    else:
        args.blockchain = True
        args.train_nodes = list(range(graph.node_label.shape[0]))
        surrogate_model = gg.gallery.nodeclas.SGCPDS(device=args.device, seed=1000).setup_graph(graph, K=1).build()
        surrogate_model.fit(args.train_nodes,
                            None,
                            verbose=args.verbose,
                            epochs=6)

        # Before attack
        attacked_model = gg.gallery.nodeclas.GCNPD(device=args.device, seed=args.seed).setup_graph(graph).build()
        attacked_model.fit(args.train_nodes,
                            None,
                            verbose=args.verbose,
                            epochs=6)
        if args.us:
            attacker = SCAPD(graph, device=args.device, seed=args.seed).process(surrogate_model)
        else:
            attacker = SGAPD(graph, device=args.device, seed=args.seed).process(surrogate_model)

    return attacked_model, attacker


def run(subgraph_type, cmd=None, p=2.0, q=0.25, alpha=0.25, verbose=True):
    data = NPZDataset(cmd.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")

    graph = data.graph
    random.seed(cmd.seed)

    splits = data.split_nodes(random_state=15)
    targets = random.sample(list(splits.test_nodes), cmd.target_nums)

    cmd.subgraph_type = subgraph_type
    cmd.p = p
    cmd.q = q
    cmd.alpha = alpha
    args = ARGS(cmd=cmd, targets=targets, splits=splits, node_attr=graph.node_attr, node_label=graph.node_label)
    print(args.device)

    attacked_model, attacker = get_attack_model(args, graph)
    if cmd.dataset not in DATASET_BLOCKCHAIN:
        res = testACC(attacked_model, attacker, args, verbose=verbose)
    else:
        res = testBlockACC(attacked_model, attacker, args, verbose=verbose)
    gc.collect()
    return res


if __name__ == '__main__':
    frame = inspect.currentframe()  # define a frame to track
    gpu_tracker = MemTracker(frame)  # define a GPU tracker
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")

    parser.add_argument("-st", "--subgraph_type", default="cluster", type=str, help="sample method")
    parser.add_argument("-sr", "--sample_ratio", default=0.05, type=float, help="ratio of sampled nodes")
    parser.add_argument("-da", "--direct_attack", default="true", type=str, help="direct attack")
    parser.add_argument("-tn", "--target_nums", default=50, type=int, help="target nums")

    parser.add_argument("--dataset", default="cora", type=str, help="dataset")
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    parser.add_argument("-p", default=7.0, type=float)
    parser.add_argument("-q", default=0.25, type=float)
    parser.add_argument("-a", "--alpha", default=0.25, type=float)
    parser.add_argument("-et", "--embed_type", default="MLP", type=str)
    parser.add_argument('-ip', '--is_phi', default="true", type=str)
    parser.add_argument('-atk', '--atk_model_type', default="GCN", type=str)

    parser.add_argument('--max_iter', default=300, type=int)
    parser.add_argument('--n_init', default=40, type=int)
    parser.add_argument('-tc', '--topk_cluster', default=1, type=int)
    parser.add_argument('-r', '--random', default="false", type=str)

    cmd = parser.parse_args()
    # cmd.embed_type = "ClusterGCN"
    # cmd.dataset = 'cora'
    # cmd.subgraph_type = "cluster"
    # cmd.atk_model_type = "MixHop"
    # cmd.random = "true"
    # cmd.topk_cluster = 3
    # cmd.direct_attack = ""
    # cmd.is_phi = "true"
    # cmd.seed = 2012
    gg.set_backend("th")
    # if cmd.atk_model_type == "MixHop":
    #     gg.set_backend("dgl")

    data = NPZDataset(cmd.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")
    graph = data.graph
    random.seed(cmd.seed)
    splits = data.split_nodes(random_state=15)
    targets = random.sample(list(splits.test_nodes), cmd.target_nums)
    args = ARGS(cmd=cmd, targets=targets, splits=splits, node_attr=graph.node_attr, node_label=graph.node_label)
    attacked_model, attacker = get_attack_model(args, graph)
    gpu_tracker.track()
    if cmd.dataset not in DATASET_BLOCKCHAIN:
        res = testACC(attacked_model, attacker, args, verbose_us=False)
    else:
        res = testBlockACC(attacked_model, attacker, args, verbose_us=False)
    gpu_tracker.track()
    gc.collect()