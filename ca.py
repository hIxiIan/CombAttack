import random
import graphgallery as gg
import numpy as np
import pandas as pd
import gc
import argparse
import inspect
import torch
import scipy.sparse as sp
import graphgallery.functional as gf
import dgl
import os
import os.path as osp
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
from utils import get_attacked_types, get_model_parms, get_pd, get_train_x, MODEL_PARAMS, DP_MODELS, accuracy, _normalize_adj, get_wrong_labels, mapCluster2GCN, _normalize_adj_simpgcn
from deeprobust.graph.defense import RGCN, SimPGCN, GCN
from fagcn import get_FAGCN
from h2gcn import get_H2GCN, sp_to_tensor
from dgl import function as fn
from tedge import get_tedge
from trans2vec import get_trans2vec
from bm_gcn import get_bmgcn, get_bmgcn_sur


def get_model(model_name, args, graph, A_V_F=None, is_embed=False, is_model=False):
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
        if is_model and args.hids is not None:
            return gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(graph).build(hids=args.hids, acts=args.acts, dropout=0, weight_decay=args.weight_decay, lr=args.lr, bias=True)
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
        if is_model and args.hids is not None:
            model = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(graph,
                                                                                            graph_transform="jaccard_detection").build(hids=args.hids, acts=args.acts, dropout=0, weight_decay=args.weight_decay, lr=args.lr, bias=True)
            model.name = "GCN_Jaccard"
            return model
        model = gg.gallery.nodeclas.GCN(device=args.device, seed=args.seed).setup_graph(graph, graph_transform="jaccard_detection").build()
        model.name = "GCN_Jaccard"
        return model
    elif model_name == "RobustGCN":
        if is_model and args.hids is not None:
            return gg.gallery.nodeclas.RobustGCN(device=args.device, seed=args.seed).setup_graph(graph).build(hids=args.hids, acts=args.acts, dropout=0, weight_decay=args.weight_decay, lr=args.lr, bias=True)
        return gg.gallery.nodeclas.RobustGCN(device=args.device, seed=args.seed).setup_graph(graph).build()

    # 异质GCN
    elif model_name == "SimPGCN":
        if is_model and args.hids is not None:
            return gg.gallery.nodeclas.SimPGCN(device=args.device, seed=args.seed).setup_graph(graph).build(hids=args.hids, acts=args.acts, dropout=0, weight_decay=args.weight_decay, lr=args.lr, bias=True)
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

    elif model_name == "FAGCN":
        model, acc = get_FAGCN(args, graph)
        model.name = "FAGCN"
        return model

    elif model_name in ["H2GCN", "H2GCN2"]:
        model, acc = get_H2GCN(args, graph, 2)
        model.name = "H2GCN2"
        return model
    elif model_name == "H2GCN1":
        model, acc = get_H2GCN(args, graph, 1)
        model.name = "H2GCN1"
        return model

    elif model_name == "BMGCN":
        assert "bc" in args.dataset, "BMGCN get_model invalid"
        if A_V_F is None:
            path = osp.abspath(osp.expanduser("~/GraphData/datasets/"))
            bmbc = np.load(''.join([path, os.sep, "bm" + args.dataset + ".npz"]), allow_pickle=True)
            A_V_F = [bmbc["A"].item(), bmbc["V"].item(), bmbc["F"].item()]
        model = get_bmgcn(args, graph, A_V_F)
        model.name = "BMGCN"
        return model

    assert False, "invalid graphgallery model:{}".format(model_name)


def get_dr_model(model_name, args, graph):
    adj = graph.adj_matrix
    features = graph.node_attr
    labels = graph.node_label
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dr_device = device
    idx_train = args.splits.train_nodes
    idx_val = args.splits.val_nodes
    idx_test = args.splits.test_nodes

    if model_name == "RobustGCN":
        attacked_model = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1,
                              nhid=32, lr=0.01, dropout=0, device=device)
    elif model_name == "SimPGCN":
        attacked_model = SimPGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1,
                              nhid=64, lr=0.01, dropout=0, weight_decay=5e-4, device=device)
    elif model_name == "GCN":
        attacked_model = GCN(nfeat=features.shape[1], nhid=64, nclass=labels.max()+1, device=device)
    else:
        assert False, "invalid deeprobust model:{}".format(model_name)
    attacked_model.name = model_name
    attacked_model.is_dr = True
    attacked_model.to(device)
    attacked_model.fit(sp.csr_matrix(features), sp.csr_matrix(adj), labels, idx_train, idx_val, train_iters=200,
                       verbose=False)
    attacked_model.eval()
    output = attacked_model.output
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return attacked_model, acc_test.item()


def get_attacked_models(atked_types, args, graph):
    attacked_models = []
    attacked_models_acc = []
    for atked_type in atked_types:
        if args.dataset != 'ogbn-arxiv' and atked_type in DP_MODELS:
            attacked_model, acc = get_dr_model(atked_type, args, graph)
            attacked_models.append(attacked_model)
            attacked_models_acc.append(acc)
            print('atked_model :{}, clean_acc: {}'.format(atked_type, acc))
            continue

        if atked_type == "BMGCN":
            attacked_model = get_model(atked_type, args, graph)
            attacked_model.is_dr = False
            acc = attacked_model.get_acc()
            attacked_models.append(attacked_model)
            attacked_models_acc.append(acc)
            print('atked_model :{}, clean_acc: {}'.format(atked_type, acc))
            continue

        attacked_model = get_model(atked_type, args, graph)
        attacked_model.is_dr = False
        attacked_model.fit(args.splits.train_nodes,
                           args.splits.val_nodes,
                           verbose=args.verbose,
                           epochs=200)
        results = attacked_model.evaluate(args.splits.test_nodes, verbose=0)
        attacked_models.append(attacked_model)
        attacked_models_acc.append(results.accuracy)
        print('atked_model :{}, clean_acc: {}'.format(atked_type, results.accuracy))
    args.attacked_models_acc = attacked_models_acc
    return attacked_models


def get_attacker(args, graph):
    if not args.blockchain or args.dataset in ["tedge", "trans2vec"] or ("bc" in args.dataset and args.bmbc_mode):
        surrogate_model = gg.gallery.nodeclas.SGC(device=args.device, seed=1000).setup_graph(graph, K=2).build()
        surrogate_model.fit(args.splits.train_nodes, args.splits.val_nodes, verbose=args.verbose, epochs=200)
        results = surrogate_model.evaluate(args.splits.test_nodes, verbose=0)
        print(f'get_attacker sur Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
        if args.us:
            attacker = SCA(graph, device=args.device, seed=args.seed).process(surrogate_model)
        else:
            attacker = SGA(graph, device=args.device, seed=args.seed).process(surrogate_model)
    else:
        args.train_nodes = list(range(graph.node_label.shape[0]))
        surrogate_model = gg.gallery.nodeclas.SGCPDS(device=args.device, seed=1000).setup_graph(graph, K=1).build()
        surrogate_model.fit(args.train_nodes, None, verbose=args.verbose, epochs=6)
        if args.us:
            attacker = SCAPD(graph, device=args.device, seed=args.seed).process(surrogate_model)
        else:
            attacker = SGAPD(graph, device=args.device, seed=args.seed).process(surrogate_model)
    return attacker


def get_atk_models(args, graph):
    if not args.blockchain:
        atked_types = get_attacked_types(args.atk_model_type)
        attacked_models = get_attacked_models(atked_types, args, graph)
        # assert len(atked_types) <= 1, 'atked_model need to be equal to 1'
        args.atked_model_ = atked_types[0]
    else:
        if args.dataset in ['tedge', 'trans2vec'] or ("bc" in args.dataset and args.bmbc_mode):
            attacked_model = gg.gallery.nodeclas.SGC(device=args.device, seed=args.seed).setup_graph(graph, K=2).build()
            attacked_model.fit(args.splits.train_nodes, args.splits.val_nodes, verbose=args.verbose, epochs=200)
            attacked_models = [attacked_model]
            args.atked_model_ = attacked_model.name
            args.attacked_models_acc = [0]
            print('{}, atked_model :{}, clean_acc: {}'.format(args.dataset, 'SGC', 0))
        else:
            attacked_model = gg.gallery.nodeclas.SGCPD(device=args.device, seed=args.seed).setup_graph(graph, K=1).build()
            attacked_model.fit(args.train_nodes, None, verbose=args.verbose, epochs=6)
            attacked_models = [attacked_model]
            args.atked_model_ = attacked_model.name
            args.attacked_models_acc = [0]
            print('blockchain, atked_model :{}, clean_acc: {}'.format('SGCPD', 0))
    return attacked_models


def get_embed_model(args, graph):
    if args.cluster_parms.mix_cluster:
        model = [get_model(model_name, args, graph, is_embed=False) for model_name in args.cluster_parms.mix_types]
    else:
        if args.embed_type in DP_MODELS:
            m, acc = get_dr_model(args.embed_type, args, graph)
            m.embed_acc = acc
            model = [m]
            print(f'get_embed_model Test accuracy {acc:.2%}')
        else:
            model = [get_model(args.embed_type, args, graph, is_embed=False)]

    for _model in model:
        if args.embed_type not in DP_MODELS:
            _model.fit(args.splits.train_nodes, args.splits.val_nodes, verbose=0, epochs=200)
            results = _model.evaluate(args.splits.test_nodes, verbose=0)
            _model.embed_acc = results.accuracy
            print(f'get_embed_model Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')

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
        model = None if args.embed_type == "ori" else get_embed_model(args, attacker.graph)
        sampler = Cluster(args.direct_attack, args.embed_type, args.targets, model, attacker.graph, args.sample_ratio,
                          attacker.logits, attacker.softmax_logits, args.cluster_parms)
        sampler.type_ = args.subgraph_type
        if args.cluster_parms.mix_cluster:
            print('mixed_type:{}, sample process end..., cost:{} min'.format(args.cluster_parms.mix_types, (time() - t1) / 60))
        else:
            print('embed_type:{}, sample process end..., cost:{} min'.format(args.embed_type, (time() - t1) / 60))
        sampler.embed_acc = 0 if args.embed_type == "ori" else np.mean([_model.embed_acc for _model in model])
    return sampler


def get_dr_results(attacked_model, attacker, args, target):
    name = attacked_model.name
    attacked_model.eval()
    # evasion
    if name == "RobustGCN":
        attacked_model.adj_norm1 = _normalize_adj(attacker.g.adj_matrix, power=-1 / 2, device=args.dr_device)
        attacked_model.adj_norm2 = _normalize_adj(attacker.g.adj_matrix, power=-1, device=args.dr_device)
        output = attacked_model.forward()
    elif name == "SimPGCN":
        adj_norm, fea = _normalize_adj_simpgcn(attacker.g.adj_matrix, attacker.g.node_attr, device=args.dr_device)
        output = attacked_model.forward(fea, adj_norm)

    eva_perturbed_label = output.max(1)[1].cpu().numpy()[target]

    # poisoning
    trainer, _ = get_dr_model(name, args, attacker.g)
    trainer.eval()
    output = trainer.output
    poi_perturbed_label = output.max(1)[1].cpu().numpy()[target]
    return eva_perturbed_label, poi_perturbed_label


def get_gf_results(attacked_model, attacker, args, target):
    name = attacked_model.name
    # evasion
    if name == "FAGCN":
        g = dgl.from_scipy(attacker.g.adj_matrix)
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)
        if args.device == "gpu":
            g = g.to("cuda")
            deg = g.in_degrees().cuda().float().clamp(min=1)
        else:
            deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        attacked_model.g = g
        for i in range(attacked_model.layer_num):
            attacked_model.layers[i].g = g
    elif name in ["H2GCN", "H2GCN1", "H2GCN2"]:
        adj = attacker.g.adj_matrix.tocoo()
        adj = sp_to_tensor(adj)
        if args.device == "gpu":
            adj = adj.to("cuda")
        attacked_model.adj = adj
        attacked_model.initialized = False
    else:
        attacked_model.setup_graph(attacker.g)
    if name == "SimPGCN":
        attacked_model.model.cache['adj_knn'] = attacked_model.cache['knn_graph']
    eva_perturbed_label = attacked_model.predict(target, transform="softmax").argmax()

    # poisoning
    trainer = get_model(name, args, attacker.g)
    trainer.fit(args.splits.train_nodes,
                args.splits.val_nodes,
                verbose=args.verbose,
                epochs=200)
    poi_perturbed_label = trainer.predict(target, transform="softmax").argmax()
    return eva_perturbed_label, poi_perturbed_label


def testACC_get_edge_flips(attacked_models, attacker, args, verbose=True, verbose_us=False):
    sampler = init_sampler(attacker, args)
    start = time()
    perturbed_edges_dict = {}
    for i, target in enumerate(args.targets):
        attacker = attacker.reset()
        try:
            if args.us:
                attacker.attack(target, sampler=sampler, verbose_us=verbose_us, direct_attack=args.direct_attack, is_topk=args.is_topk)
            else:
                attacker.attack(target, verbose_us=False, direct_attack=args.direct_attack)
        except AssertionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        except PermissionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        perturbed_edges_dict[target] = attacker.adj_flips
    end = time()
    cost = (end - start) / 60
    embed_acc = sampler.embed_acc if sampler is not None else 0
    if args.subgraph_type != "cluster":
        print('subgraph:{}, p:{}, q:{}, alpha:{}'.format(args.subgraph_type, args.p, args.q, args.alpha))
    else:
        if args.cluster_parms.mix_cluster:
            print('mix_types:{}, embed_acc:{}'.format(args.cluster_parms.mix_types, embed_acc))
        else:
            print('embed_type:{}, embed_acc:{}'.format(args.embed_type, embed_acc))
    print('testACC end, cost time: {} min'.format(cost))
    return perturbed_edges_dict, []


def testACC(attacked_models, attacker, args, verbose=True, verbose_us=False):
    sampler = init_sampler(attacker, args)
    if sampler is not None:
        targets_labels_pred = sampler.cluster_label_pred[args.targets]
        # surrogate model predicted labels
        sur_labels = sampler.sur_labels
        # second largest pro class , computed by surrogate logits and ground truth
        wrong_labels = sampler.wrong_labels
        # get the max pro class of added cluster nodes
        add_gcn_labels, add_gcn_labels_rate = mapCluster2GCN(args.targets, sur_labels, sampler.added_edges)
        add_clusterIsMisClassifiedLabel = wrong_labels == add_gcn_labels
        if verbose:
            print('cluster: targets labels:{}, {}'.format(set(targets_labels_pred),
                                                      [(l, (targets_labels_pred == l).sum()) for l in
                                                       set(targets_labels_pred)]))
            print('added_nodes labels: {}'.format(add_gcn_labels))
            print('added_nodes labels pro: {}'.format(add_gcn_labels_rate))
            print('wrong_labels: {}'.format(wrong_labels))
            print('add_clusterIsMisClassifiedLabel:{}'.format(add_clusterIsMisClassifiedLabel))

    start = time()
    eva_res = {}
    # eva_res_wl = {}
    poi_res = {}
    # poi_res_wl = {}
    original_predicts = {}

    for attacked_model in attacked_models:
        name = attacked_model.name
        if attacked_model.is_dr:
            sur_preds = gf.get('softmax')(attacked_model.predict().detach().cpu().numpy())
            original_predicts[name] = sur_preds[args.targets]
        else:
            sur_preds = attacked_model.predict(np.arange(len(args.node_label)), transform="softmax")
            original_predicts[name] = sur_preds[args.targets]
        eva_res[name] = np.zeros(len(args.targets)).astype('bool')
        # eva_res_wl[name] = np.zeros(len(args.targets)).astype('bool')
        poi_res[name] = np.zeros(len(args.targets)).astype('bool')
        # poi_res_wl[name] = np.zeros(len(args.targets)).astype('bool')

    cost_targets = 0.
    perturbed_edges_dict = {}
    for i, target in enumerate(args.targets):
        attacker = attacker.reset()
        start_i = time()
        try:
            if args.us:
                attacker.attack(target, sampler=sampler, verbose_us=verbose_us, direct_attack=args.direct_attack, is_topk=args.is_topk)
            else:
                attacker.attack(target, verbose_us=False, direct_attack=args.direct_attack)
        except AssertionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        except PermissionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        end_i = time()
        cost_targets = end_i - start_i
        perturbed_edges_dict[target] = attacker.adj_flips
        # After attack
        # wrong_label = int(attacker.wrong_label[0])

        for ai in range(len(attacked_models)):
            attacked_model = attacked_models[ai]
            name = attacked_model.name
            true_label = original_predicts[name][i].argmax()
            if attacked_model.is_dr:
                eva_perturbed_label, poi_perturbed_label = get_dr_results(attacked_model, attacker, args, target)
            else:
                eva_perturbed_label, poi_perturbed_label = get_gf_results(attacked_model, attacker, args, target)

            if eva_perturbed_label != true_label:
                eva_res[name][i] = True
                # if eva_perturbed_label == wrong_label:
                #     eva_res_wl[name][i] = True

            if poi_perturbed_label != true_label:
                poi_res[name][i] = True
                # if perturbed_label == wrong_label:
                #     poi_res_wl[name][i] = True

            if verbose:
                print('###################')
                print('name:{}, iter: {}, attack target node {}, get subgraph cost:{}, attack cost: {} min'.format(name, i, target, 0, (end_i - start_i) / 60))
                # print('deleted_edges.shape:{}, added_edges.shape:{}, total_nodes:{}'.format(attacker._hop_ratio, attacker._hop_length, attacker._walk_length))
                # print('wrong_ratio:{}, wrong_length:{}'.format(attacker._wrong_ratio, attacker._wrong_length))
                print('added_edges.shape:{}, added_edges:{}'.format(len(attacker.added_edges), attacker.added_edges))
                print('deleted_edges.shape:{}, deleted_edges:{}'.format(len(attacker.non_added_edges), attacker.non_added_edges))
                print('original_predict, true_label: {}, true_label_prob: {}'.format(true_label, original_predicts[name][i][true_label]))

                if np.isnan(original_predicts[name][i][true_label]):
                    print('!!!!!!!!!!!!!!!!!!!!!!nan!!!!!!!:{}'.format(original_predicts[name]))

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
        if sampler is not None:
            is_add = add_clusterIsMisClassifiedLabel
            c_is_add_eva_asr = c_is_add_poi_asr = c_isn_add_eva_asr = c_isn_add_poi_asr = 0
            if np.any(is_add):
                c_is_add_eva_asr = eva_res[name][is_add].mean()
                c_is_add_poi_asr = poi_res[name][is_add].mean()
            if np.any(~is_add):
                c_isn_add_eva_asr = eva_res[name][~is_add].mean()
                c_isn_add_poi_asr = poi_res[name][~is_add].mean()
            print('attack {} model, farthest_cluster is the misclassified class, nodes nums:{}, eva_asr:{}'.format(name, len(eva_res[name][is_add]), c_is_add_eva_asr))
            print('attack {} model, farthest_cluster is the misclassified class, nodes nums:{}, poi_asr:{}'.format(name, len(poi_res[name][is_add]), c_is_add_poi_asr))
            print('attack {} model, farthest_cluster isn\'t the misclassified class, nodes nums:{}, eva_asr:{}'.format(name, len(eva_res[name][~is_add]), c_isn_add_eva_asr))
            print('attack {} model, farthest_cluster isn\'t the misclassified class, nodes nums:{}, poi_asr:{}'.format(name, len(poi_res[name][~is_add]), c_isn_add_poi_asr))

    embed_acc = sampler.embed_acc if sampler is not None else 0
    cluster_cost_time = sampler.cluster_cost_time if sampler is not None else 0
    if args.subgraph_type != "cluster":
        print('subgraph:{}, p:{}, q:{}, alpha:{}'.format(args.subgraph_type, args.p, args.q, args.alpha))
    else:
        if args.cluster_parms.mix_cluster:
            print('mix_types:{}, embed_acc:{}'.format(args.cluster_parms.mix_types, embed_acc))
        else:
            print('embed_type:{}, embed_acc:{}'.format(args.embed_type, embed_acc))
    print('testACC end, cost time: {} min'.format(cost))
    return perturbed_edges_dict, [[attacked_model.name, eva_asr[attacked_model.name], poi_asr[attacked_model.name], cost, embed_acc, args.attacked_models_acc[i], cost_targets / len(args.targets), cluster_cost_time] for i, attacked_model in enumerate(attacked_models)]


def testBlockACC_get_edge_flips(attacked_models, attacker, args, verbose=True, verbose_us=False):
    if args.is_phi:
        if "bc" in args.dataset and not args.bmbc_mode:
            attacked_model = attacked_models[0]
            original_predict, _ = get_pd(attacked_model, args)
        else:
            if args.dataset == "tedge":
                original_predict = get_tedge(args) # lcc
            elif args.dataset == "trans2vec":
                original_predict = get_trans2vec(args)
            elif "bc" in args.dataset and args.bmbc_mode:
                original_predict = get_bmgcn_sur(args)
        surrogate_phishing_targets = np.where(original_predict == 1)[0]
        true_phishing_targets = np.where(args.node_label == 1)[0]
        ori_targets = np.intersect1d(surrogate_phishing_targets, true_phishing_targets)
        if len(ori_targets) <= args.target_nums:
            print('len(ori_targets): {}, args.target_nums:{}'.format(len(ori_targets), args.target_nums))
            args.target_nums = len(ori_targets)
            args.targets = list(ori_targets)
        else:
            args.targets = random.sample(list(ori_targets), args.target_nums)

        print('attack {} phishing nodes, total {} phishing nodes, total true phishing nodes:{}, total surrogate_phishing_nodes:{}'.format(
            len(args.targets), len(ori_targets), len(true_phishing_targets), len(surrogate_phishing_targets)))
    else:
        print('attack phishing or non-phishing nodes')

    if len(args.targets) == 0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!target is None!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    sampler = init_sampler(attacker, args)
    start = time()
    perturbed_edges_dict = {}
    for i, target in enumerate(args.targets):
        attacker = attacker.reset()
        try:
            if args.us:
                attacker.attack(target, sampler=sampler, verbose_us=verbose_us, direct_attack=args.direct_attack,
                                blockchain=args.blockchain, is_topk=args.is_topk)
            else:
                attacker.attack(target, verbose_us=False, direct_attack=args.direct_attack, blockchain=args.blockchain)
        except AssertionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        except PermissionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        perturbed_edges_dict[target] = attacker.adj_flips
    end = time()
    cost = (end - start) / 60
    embed_acc = sampler.embed_acc if sampler is not None else 0
    if args.subgraph_type != "cluster":
        print('subgraph:{}, p:{}, q:{}, alpha:{}'.format(args.subgraph_type, args.p, args.q, args.alpha))
    else:
        if args.cluster_parms.mix_cluster:
            print('mix_types:{}, embed_acc:{}'.format(args.cluster_parms.mix_types, embed_acc))
        else:
            print('embed_type:{}, embed_acc:{}'.format(args.embed_type, embed_acc))
    print('testBlockACC end, cost time: {} min'.format(cost))
    print('embed_acc:{}'.format(embed_acc))

    return perturbed_edges_dict, []


def testBlockACC(attacked_models, attacker, args, verbose=True, verbose_us=False):
    assert args.dataset not in ["tedge", "trans2vec"], "testBlockACC tedge, trans2vec error"
    attacked_model = attacked_models[0]
    original_predict, lgb_model = get_pd(attacked_model, args)
    if args.is_phi:
        surrogate_phishing_targets = np.where(original_predict == 1)[0]
        true_phishing_targets = np.where(args.node_label == 1)[0]
        ori_targets = np.intersect1d(surrogate_phishing_targets, true_phishing_targets)
        if len(ori_targets) <= args.target_nums:
            print('len(ori_targets): {}, args.target_nums:{}'.format(len(ori_targets), args.target_nums))
            args.target_nums = len(ori_targets)
            args.targets = list(ori_targets)
        else:
            args.targets = random.sample(list(ori_targets), args.target_nums)
        print('attack {} phishing nodes, total true phishing nodes:{}, total surrogate_phishing_nodes:{}'.format(
            len(args.targets), len(true_phishing_targets), len(surrogate_phishing_targets)))
    else:
        print('attack phishing or non-phishing nodes')

    sampler = init_sampler(attacker, args)
    eva_res = np.zeros(len(args.targets)).astype('bool')
    poi_res = np.zeros(len(args.targets)).astype('bool')
    start = time()
    cost_targets = 0.0
    perturbed_edges_dict = {}
    for i, target in enumerate(args.targets):
        attacker = attacker.reset()
        start_i = time()
        try:
            if args.us:
                attacker.attack(target, sampler=sampler, verbose_us=verbose_us, direct_attack=args.direct_attack, blockchain=args.blockchain, is_topk=args.is_topk)
            else:
                attacker.attack(target, verbose_us=False, direct_attack=args.direct_attack, blockchain=args.blockchain)
        except AssertionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        except PermissionError as e:
            print('iter: {}. ###############, error: {}'.format(i, repr(e)))
        end_i = time()
        cost_targets = end_i - start_i
        perturbed_edges_dict[target] = attacker.adj_flips
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
        if args.cluster_parms.mix_cluster:
            print('mix_types:{}, embed_acc:{}'.format(args.cluster_parms.mix_types, embed_acc))
        else:
            print('embed_type:{}, embed_acc:{}'.format(args.embed_type, embed_acc))
    print('testBlockACC end, cost time: {} min'.format(cost))
    print('embed_acc:{}'.format(embed_acc))

    return perturbed_edges_dict, [[attacked_model.name, eva_asr, poi_asr, cost, embed_acc, args.attacked_models_acc[0], cost_targets / len(args.targets), cluster_cost_time]]


def run(subgraph_type, cmd=None, p=2.0, q=0.25, alpha=0.25, verbose=True):
    graph, splits, targets = get_dataset(cmd)
    cmd.subgraph_type = subgraph_type
    cmd.p = p
    cmd.q = q
    cmd.alpha = alpha
    args = ARGS(cmd=cmd, targets=targets, splits=splits, graph=graph)
    print(args.device)
    attacker = get_attacker(args, graph)
    if cmd.target_mode == "correct_sur_labels":
        gf.random_seed(cmd.seed, gg.backend())
        candidates = np.array(splits.test_nodes)
        sur_labels = np.array([lo.argmax() for lo in attacker.softmax_logits])
        ground_truths = graph.node_label
        predict_suc_idx = sur_labels[candidates] == ground_truths[candidates]
        test_nodes = candidates[predict_suc_idx]
        assert cmd.target_nums <= len(test_nodes), "the number of target nodes is larger than predicted suc nodes"
        targets = random.sample(list(test_nodes), cmd.target_nums)
        args.targets = targets
    if not args.blockchain:
        if args.edge_flips:
            perturbed_edges_dict, res = testACC_get_edge_flips(None, attacker, args, verbose=verbose)
        else:
            attacked_models = get_atk_models(args, graph)
            perturbed_edges_dict, res = testACC(attacked_models, attacker, args, verbose=verbose)
    else:
        attacked_models = None
        if "bc" in args.dataset and not args.bmbc_mode:
            attacked_models = get_atk_models(args, graph)
        if args.edge_flips:
            perturbed_edges_dict, res = testBlockACC_get_edge_flips(attacked_models, attacker, args, verbose=verbose)
        else:
            perturbed_edges_dict, res = testBlockACC(attacked_models, attacker, args, verbose=verbose)
    # print(res)
    gc.collect()
    return perturbed_edges_dict, res


def get_dataset(cmd):
    data = NPZDataset(cmd.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")
    graph = data.graph
    gf.random_seed(cmd.seed, gg.backend())
    splits = data.split_nodes(random_state=15)
    targets = random.sample(list(splits.test_nodes), cmd.target_nums)
    return graph, splits, targets


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
    parser.add_argument("-et", "--embed_type", default="GCN2", type=str)
    parser.add_argument('-ip', '--is_phi', default="true", type=str)
    parser.add_argument('-atk', '--atk_model_type', default="GCN", type=str)

    parser.add_argument('--max_iter', default=300, type=int)
    parser.add_argument('--n_init', default=40, type=int)
    parser.add_argument('-tc', '--topk_cluster', default=1, type=int)
    parser.add_argument('-r', '--random', default="false", type=str)
    parser.add_argument('-la', '--lay_act', default="layer", type=str)
    parser.add_argument('-lac', '--lay_act_cnt', default=999, type=int)
    parser.add_argument('-dt', '--distance_type', default="euclidean", type=str)
    parser.add_argument('-tk', '--is_topk', default="false", type=str)
    parser.add_argument('-ef', '--edge_flips', default="false", type=str)
    parser.add_argument('-tm', '--target_mode', default="sur_labels", type=str)
    parser.add_argument('-mc', '--mix_cluster', default="false", type=str)
    parser.add_argument('-mts', '--mix_types', default="MLP,SGC2", type=str)
    parser.add_argument("--test_mode", default="-1", type=str)
    parser.add_argument("--features_mode", default="ori", type=str)
    parser.add_argument("--deg_limit", default=2, type=int)
    parser.add_argument("--sur_label_pro_limit", default=0.9, type=float)
    parser.add_argument("--features_file", default="", type=str)
    parser.add_argument("--train_size", default=0.5, type=float)
    parser.add_argument("--trans2vec_model", default="OCSVM", type=str)
    parser.add_argument('--bmbc_mode', default="false", type=str)
    parser.add_argument('--T', type=int, default=100)

    cmd = parser.parse_args()
    cmd.hids, cmd.acts, cmd.weight_decay, cmd.lr = None, None, None, None
    gg.set_backend("th")
    graph, splits, targets = get_dataset(cmd)
    args = ARGS(cmd=cmd, targets=targets, splits=splits, graph=graph)
    attacker = get_attacker(args, graph)
    if cmd.target_mode == "correct_sur_labels":
        gf.random_seed(cmd.seed, gg.backend())
        candidates = np.array(splits.test_nodes)
        sur_labels = np.array([lo.argmax() for lo in attacker.softmax_logits])
        ground_truths = graph.node_label
        predict_suc_idx = sur_labels[candidates] == ground_truths[candidates]
        test_nodes = candidates[predict_suc_idx]
        assert cmd.target_nums <= len(test_nodes), "the number of target nodes is larger than predicted suc nodes"
        targets = random.sample(list(test_nodes), cmd.target_nums)
        args.targets = targets

    # gpu_tracker.track()
    if not args.blockchain:
        if args.edge_flips:
            perturbed_edges_dict, res = testACC_get_edge_flips(None, attacker, args, verbose_us=False)
        else:
            attacked_models = get_atk_models(args, graph)
            perturbed_edges_dict, res = testACC(attacked_models, attacker, args, verbose_us=False)
    else:
        attacked_models = get_atk_models(args, graph)
        if args.edge_flips:
            perturbed_edges_dict, res = testBlockACC_get_edge_flips(attacked_models, attacker, args, verbose_us=False)
        else:
            perturbed_edges_dict, res = testBlockACC(attacked_models, attacker, args, verbose_us=False)
    # print(perturbed_edges_dict)
    # print(res)
    # gpu_tracker.track()
    gc.collect()