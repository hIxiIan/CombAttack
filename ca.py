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
from utils import get_attacked_types, get_model_parms, get_pd, get_train_x, MODEL_PARAMS


def get_model(model_name, args, graph, is_embed=False):
    # GCN
    if model_name == "SGC":
        return gg.gallery.nodeclas.SGC(device=args.device, seed=args.seed).setup_graph(graph, K=2).build()
    elif model_name == "SGC2":
        if args.hids is not None:
            return gg.gallery.nodeclas.SGC2(device=args.device, seed=args.seed).setup_graph(graph, K=2).build(hids=args.hids, acts=args.acts, dropout=0, weight_decay=args.weight_decay, lr=args.lr, bias=True)

        if is_embed and args.dataset in MODEL_PARAMS:
            hids, acts, weight_decay, lr = get_model_parms(args.dataset, model_name, args.atked_model_)
            return gg.gallery.nodeclas.SGC2(device=args.device, seed=args.seed).setup_graph(graph, K=2).build(hids=hids, acts=acts, dropout=0, weight_decay=weight_decay, lr=lr, bias=True)
        return gg.gallery.nodeclas.SGC2(device=args.device, seed=args.seed).setup_graph(graph, K=2).build()
    elif model_name == "GCN":
        return gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(graph).build()
    if model_name == "GCN2":
        if args.hids is not None:
            return gg.gallery.nodeclas.GCN2(device=args.device, seed=args.seed).setup_graph(graph).build(hids=args.hids, acts=args.acts, dropout=0, weight_decay=args.weight_decay, lr=args.lr, bias=True)

        if is_embed and args.dataset in MODEL_PARAMS:
            hids, acts, weight_decay, lr = get_model_parms(args.dataset, model_name, args.atked_model_)
            return gg.gallery.nodeclas.GCN2(device=args.device, seed=args.seed).setup_graph(graph).build(hids=hids, acts=acts, dropout=0, weight_decay=weight_decay, lr=lr, bias=True)
        return gg.gallery.nodeclas.GCN2(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif model_name == "FastGCN":
        if args.hids is not None:
            return gg.gallery.nodeclas.FastGCN(device=args.device, seed=args.seed).setup_graph(graph).build(hids=args.hids, acts=args.acts, dropout=0, weight_decay=args.weight_decay, lr=args.lr, bias=True)

        if is_embed and args.dataset in MODEL_PARAMS:
            hids, acts, weight_decay, lr = get_model_parms(args.dataset, model_name, args.atked_model_)
            return gg.gallery.nodeclas.FastGCN(device=args.device, seed=args.seed).setup_graph(graph).build(hids=hids, acts=acts, dropout=0, weight_decay=weight_decay, lr=lr, bias=True)
        return gg.gallery.nodeclas.FastGCN(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif model_name == "ClusterGCN":
        return gg.gallery.nodeclas.ClusterGCN(device=args.device, seed=args.seed).setup_graph(graph, num_clusters=10).build()

    # 鲁棒GCN
    elif model_name == "GCN_Jaccard":
        model = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(graph, graph_transform="jaccard_detection").build()
        model.name = "GCN_Jaccard"
        return model
    elif model_name == "RobustGCN":
        return gg.gallery.nodeclas.RobustGCN(device=args.device, seed=args.seed).setup_graph(graph).build()

    # 异质GCN
    elif model_name == "SimPGCN":
        return gg.gallery.nodeclas.SimPGCN(device=args.device, seed=args.seed).setup_graph(graph).build()

    # 空域GCN
    elif model_name == "GraphSAGE":
        return gg.gallery.nodeclas.GraphSAGE(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif model_name == "GAT":
        return gg.gallery.nodeclas.GAT(device=args.device, seed=args.seed).setup_graph(graph).build()

    # NN
    elif model_name == "MLP":
        if args.hids is not None:
            return gg.gallery.nodeclas.MLP(device=args.device, seed=args.seed).setup_graph(graph).build(hids=args.hids, acts=args.acts, dropout=0, weight_decay=args.weight_decay, lr=args.lr, bias=True)

        if is_embed and args.dataset in MODEL_PARAMS:
            hids, acts, weight_decay, lr = get_model_parms(args.dataset, model_name, args.atked_model_)
            return gg.gallery.nodeclas.MLP(device=args.device, seed=args.seed).setup_graph(graph).build(hids=hids, acts=acts, dropout=0, weight_decay=weight_decay, lr=lr, bias=True)
        return gg.gallery.nodeclas.MLP(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif model_name == "GraphMLP":
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
        return gg.gallery.nodeclas.GraphMLP(device=args.device, seed=args.seed).setup_graph(graph).build(tau=tau, alpha=alpha)
    elif model_name == "PPNP":
        return gg.gallery.nodeclas.PPNP(device=args.device, seed=args.seed).setup_graph(graph).build()
    elif model_name == "APPNP":
        return gg.gallery.nodeclas.APPNP(device=args.device, seed=args.seed).setup_graph(graph).build()

    # 随机游走
    elif model_name == "DW":
        return gg.gallery.embedding.DeepWalk()
    elif model_name == "N2V":
        return gg.gallery.embedding.Node2Vec(p=args.p, q=args.q)
    elif model_name == "BANE":
        return gg.gallery.embedding.BANE()

    # dgl backend
    elif model_name == "MixHop":
        return gg.gallery.nodeclas.MixHop(device=args.device, seed=args.seed).setup_graph(graph).build()
    return None


def get_attacked_models(atked_types, args, graph):
    attacked_models = []
    for atked_type in atked_types:
        attacked_model = get_model(atked_type, args, graph)
        attacked_model.fit(args.splits.train_nodes,
                           args.splits.val_nodes,
                           verbose=args.verbose,
                           epochs=100)
        attacked_models.append(attacked_model)
    return attacked_models


def get_attacker_attacked_models(args, graph):
    if not args.blockchain:
        surrogate_model = gg.gallery.nodeclas.SGC(device=args.device, seed=1000).setup_graph(graph, K=2).build()
        surrogate_model.fit(args.splits.train_nodes, args.splits.val_nodes, verbose=args.verbose, epochs=100)

        # Before attack
        atked_types = get_attacked_types(args.atk_model_type)
        attacked_models = get_attacked_models(atked_types, args, graph)
        # assert len(atked_types) <= 1, 'atked_model need to be equal to 1'

        # 限定只攻击一个模型，认为控制args.atk_model_type唯一才有用
        args.atked_model_ = atked_types[0]
        if args.us:
            attacker = SCA(graph, device=args.device, seed=args.seed).process(surrogate_model)
        else:
            attacker = SGA(graph, device=args.device, seed=args.seed).process(surrogate_model)
    else:
        args.train_nodes = list(range(graph.node_label.shape[0]))
        surrogate_model = gg.gallery.nodeclas.SGCPDS(device=args.device, seed=1000).setup_graph(graph, K=1).build()
        surrogate_model.fit(args.train_nodes, None, verbose=args.verbose, epochs=6)

        # Before attack
        attacked_model = gg.gallery.nodeclas.GCNPD(device=args.device, seed=args.seed).setup_graph(graph).build()
        attacked_model.fit(args.train_nodes, None, verbose=args.verbose, epochs=6)
        attacked_models = [attacked_model]
        if args.us:
            attacker = SCAPD(graph, device=args.device, seed=args.seed).process(surrogate_model)
        else:
            attacker = SGAPD(graph, device=args.device, seed=args.seed).process(surrogate_model)

    return attacked_models, attacker


def get_embed_model(args, graph):
    model = get_model(args.embed_type, args, graph, is_embed=True)
    if args.embed_type not in ["DW", 'N2V', 'BANE']:
        model.fit(args.splits.train_nodes, args.splits.val_nodes, verbose=0, epochs=100)
        results = model.evaluate(args.splits.test_nodes, verbose=0)
        print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
    else:
        if args.embed_type in ["DW", "N2V"]:
            model.fit(graph.adj_matrix)
        elif args.embed_type == "BANE":
            model.fit(graph.adj_matrix, graph.node_attr)
        results = model.evaluate_nodeclas(graph.node_label, args.splits.train_nodes, args.splits.test_nodes)
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


def testACC(attacked_models, attacker, args, verbose=True, verbose_us=False):
    sampler = init_sampler(attacker, args)
    start = time()
    eva_res = {}
    # eva_res_wl = {}
    poi_res = {}
    # poi_res_wl = {}
    original_predicts = {}
    for attacked_model in attacked_models:
        name = attacked_model.name
        original_predicts[name] = attacked_model.predict(args.targets, transform="softmax")
        eva_res[name] = np.zeros(len(args.targets)).astype('bool')
        # eva_res_wl[name] = np.zeros(len(args.targets)).astype('bool')
        poi_res[name] = np.zeros(len(args.targets)).astype('bool')
        # poi_res_wl[name] = np.zeros(len(args.targets)).astype('bool')

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
        # wrong_label = int(attacker.wrong_label[0])

        for ai in range(len(attacked_models)):
            attacked_model = attacked_models[ai]

            # evasion
            attacked_model.setup_graph(attacker.g)
            name = attacked_model.name
            if name == "SimPGCN":
                attacked_model.model.cache['adj_knn'] = attacked_model.cache['knn_graph']
            true_label = original_predicts[name][i].argmax()
            eva_perturbed_label = attacked_model.predict(target, transform="softmax").argmax()
            if eva_perturbed_label != true_label:
                eva_res[name][i] = True
                # if eva_perturbed_label == wrong_label:
                #     eva_res_wl[name][i] = True

            # poisoning
            trainer = get_model(name, args, attacker.g)
            trainer.fit(args.splits.train_nodes,
                              args.splits.val_nodes,
                              verbose=args.verbose,
                              epochs=100)
            perturbed_label = trainer.predict(target, transform="softmax").argmax()
            if perturbed_label != true_label:
                poi_res[name][i] = True
                # if perturbed_label == wrong_label:
                #     poi_res_wl[name][i] = True

            if verbose:
                print('###################')
                print('iter: {}, attack target node {}, get subgraph cost:{}, attack cost: {} min'.format(i, target, 0, (end_i - start_i) / 60))
                # print('deleted_edges.shape:{}, added_edges.shape:{}, total_nodes:{}'.format(attacker._hop_ratio, attacker._hop_length, attacker._walk_length))
                # print('wrong_ratio:{}, wrong_length:{}'.format(attacker._wrong_ratio, attacker._wrong_length))
                print('added_edges.shape:{}, added_edges:{}'.format(len(attacker.added_edges), attacker.added_edges))
                print('deleted_edges.shape:{}, deleted_edges:{}'.format(len(attacker.non_added_edges), attacker.non_added_edges))
                print('original_predict, true_label: {}, true_label_prob: {}'.format(true_label, original_predicts[name][i][true_label]))

                print('#####evasion')
                # print('eva_predict, true_label: {}, true_label_prob: {}'.format(true_label, eva_predict[true_label]))
                # print('eva_predict, perturbed_label: {}, perturbed_label_prob:{}'.format(eva_perturbed_label, eva_predict[eva_perturbed_label]))
                # print('eva_predict, wrong_label: {}, wrong_label_prob:{}'.format(wrong_label, eva_predict[wrong_label]))
                # print('target node {}, mislead to label: {}, max_label_prob_sub_perturbed_label_prob: {}'.format(
                #     target, eva_perturbed_label, eva_max_label_prob_sub_perturbed_label_prob))
                # print('current eva_asr: {}, eva_asr_wl: {}'.format(eva_res[:i + 1].sum() / (i + 1), eva_res_wl[:i + 1].sum() / (i + 1)))
                print('current eva_asr: {}'.format(eva_res[name][:i + 1].sum() / (i + 1)))

                print('#####poisoning')
                # print('perturbed_predict, true_label: {}, true_label_prob: {}'.format(true_label, perturbed_predict[true_label]))
                # print('perturbed_predict, perturbed_label: {}, perturbed_label_prob:{}'.format(perturbed_label, perturbed_predict[perturbed_label]))
                # print('perturbed_predict, wrong_label: {}, wrong_label_prob:{}'.format(wrong_label, perturbed_predict[wrong_label]))
                # print('target node {}, mislead to label: {}, max_label_prob_sub_perturbed_label_prob: {}'.format(
                #     target, perturbed_label, max_label_prob_sub_perturbed_label_prob))
                # print('current poi_asr: {}, poi_asr_wl: {}'.format(poi_res[:i + 1].sum() / (i + 1), poi_res_wl[:i + 1].sum() / (i + 1)))
                print('current poi_asr: {}'.format(poi_res[name][:i + 1].sum() / (i + 1)))
                print('\n\n\n')

    end = time()
    cost = (end - start) / 60
    eva_asr = {}
    # eva_asr_wl = {}
    poi_asr = {}
    # poi_asr_wl = {}
    for attacked_model in attacked_models:
        name = attacked_model.name
        eva_asr[name] = eva_res[name].sum() / len(eva_res[name])
        # eva_asr_wl[name] = eva_res_wl[name].sum() / len(eva_res_wl[name])
        poi_asr[name] = poi_res[name].sum() / len(poi_res[name])
        # poi_asr_wl[name] = poi_res_wl[name].sum() / len(poi_res_wl[name])
        # print('eva_asr: {}, eva_asr_wl: {}'.format(eva_asr, eva_asr_wl))
        # print('poi_asr: {}, poi_asr_wl: {}'.format(poi_asr, poi_asr_wl))
        print('attack {} model, eva_asr: {}'.format(name, eva_asr[name]))
        print('attack {} model, poi_asr: {}'.format(name, poi_asr[name]))

    embed_acc = sampler.embed_acc if sampler is not None else 0
    cluster_cost_time = sampler.cluster_cost_time if sampler is not None else 0
    if args.subgraph_type != "cluster":
        print('subgraph:{}, p:{}, q:{}, alpha:{}'.format(args.subgraph_type, args.p, args.q, args.alpha))
    else:
        print('embed_type:{}, embed_acc:{}'.format(args.embed_type, embed_acc))
    print('testACC end, cost time: {} min'.format(cost))
    return [[attacked_model.name, eva_asr[attacked_model.name], poi_asr[attacked_model.name], cost, embed_acc, cost_targets / len(args.targets), cluster_cost_time] for attacked_model in attacked_models]


def testBlockACC(attacked_models, attacker, args, verbose=True, verbose_us=False):
    attacked_model = attacked_models[0]
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
            # print('deleted_edges.shape:{}, added_edges.shape:{}, total_nodes:{}'.format(attacker._hop_ratio, attacker._hop_length, attacker._walk_length))
            # print('wrong_ratio:{}, wrong_length:{}'.format(attacker._wrong_ratio, attacker._wrong_length))
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

    return [[eva_asr, poi_asr, cost, embed_acc, cost_targets / len(args.targets), cluster_cost_time]]


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

    attacked_models, attacker = get_attacker_attacked_models(args, graph)
    if not args.blockchain:
        res = testACC(attacked_models, attacker, args, verbose=verbose)
    else:
        res = testBlockACC(attacked_models, attacker, args, verbose=verbose)
    gc.collect()
    return res


if __name__ == '__main__':
    # frame = inspect.currentframe()  # define a frame to track
    # gpu_tracker = MemTracker(frame)  # define a GPU tracker
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
    cmd.hids = None
    cmd.acts = None
    cmd.weight_decay = None
    cmd.lr = None
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
    attacked_models, attacker = get_attacker_attacked_models(args, graph)
    # gpu_tracker.track()
    if not args.blockchain:
        res = testACC(attacked_models, attacker, args, verbose_us=False)
    else:
        res = testBlockACC(attacked_models, attacker, args, verbose_us=False)
    # gpu_tracker.track()
    gc.collect()