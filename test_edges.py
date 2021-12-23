import argparse
import os
import pandas as pd
import numpy as np
import torch
import graphgallery as gg
import graphgallery.functional as gf
import scipy.sparse as sp
from ca import get_model, get_dr_model
from graphgallery.datasets import NPZDataset
from time import strftime, localtime
from utils import get_datasets, _normalize_adj, _normalize_adj_simpgcn


ds = ["GCN", "SGC", "SimPGCN", "GCN_Jaccard", "FastGCN"]


def load_json(filename):
    if 'npy' not in filename:
        filename += '.npy'
    return np.load(filename, allow_pickle=True).item()


def get_models(models):
    if len(models) <= 0:
        return ds
    return models.split(",")


def print_args(args):
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)


def save_results(times, results, filename):
    tdf = None
    for i in range(times):
        df = results[i]
        if i == 0:
            tdf = df
        else:
            tdf += df
    tdf /= times
    filename = "_".join(filename.split("_")[:-1]) + '_total.csv'
    print(filename)
    tdf.to_csv(filename)
    print(tdf)


def get_adj_flips(edge_flips):
    flips = edge_flips
    if flips is None or len(flips) == 0:
        return None

    if isinstance(flips, dict):
        flips = list(flips.keys())

    return np.asarray(flips, dtype="int64")


def A(graph, edge_flips):
    adj_flips = get_adj_flips(edge_flips)
    if adj_flips is not None:
        modified_adj = gf.flip_adj(graph.adj_matrix, adj_flips)
    else:
        modified_adj = graph.adj_matrix

    adj = modified_adj

    if gf.is_tensor(adj):
        adj = gf.tensoras(adj)

    if isinstance(adj, np.ndarray):
        adj = sp.csr_matrix(adj)
    elif sp.isspmatrix(adj):
        adj = adj.tocsr(copy=False)
    else:
        raise TypeError(adj)

    return adj


def get_perturbed_graph(graph, edge_flips):
    graph = graph.copy()
    adj = A(graph, edge_flips)
    updates = dict(adj_matrix=adj)
    graph.update(**updates)

    return graph


def get_gf_results(name, graph, perturbed_graph, args, target, is_eva, is_poi):
    eva_model = get_model(name, args, graph)
    eva_model.fit(args.splits.train_nodes, args.splits.val_nodes, verbose=args.verbose, epochs=200)
    true_label = eva_model.predict(target, transform="softmax").argmax()
    eva_perturbed_label = None
    poi_perturbed_label = None

    # evasion
    if is_eva:
        eva_model.setup_graph(perturbed_graph)
        if name == "SimPGCN":
            eva_model.model.cache['adj_knn'] = eva_model.cache['knn_graph']
        eva_perturbed_label = eva_model.predict(target, transform="softmax").argmax()

    # poisoning
    if is_poi:
        poi_model = get_model(name, args, perturbed_graph)
        poi_model.fit(args.splits.train_nodes, args.splits.val_nodes, verbose=args.verbose, epochs=200)
        poi_perturbed_label = poi_model.predict(target, transform="softmax").argmax()

    eva_asr = true_label != eva_perturbed_label if eva_perturbed_label is not None else False
    poi_asr = true_label != poi_perturbed_label if poi_perturbed_label is not None else False

    return eva_asr, poi_asr


def get_dr_results(name, graph, perturbed_graph, args, target, is_eva, is_poi):
    eva_model, _ = get_dr_model(name, args, graph)
    eva_model.eval()
    true_label = eva_model.output.max(1)[1].cpu().numpy()[target]
    eva_perturbed_label = None
    poi_perturbed_label = None

    # evasion
    if is_eva:
        if name == "RobustGCN":
            eva_model.adj_norm1 = _normalize_adj(perturbed_graph.adj_matrix, power=-1 / 2, device=args.dr_device)
            eva_model.adj_norm2 = _normalize_adj(perturbed_graph.adj_matrix, power=-1, device=args.dr_device)
            output = eva_model.forward()
        elif name == "SimPGCN":
            adj_norm, fea = _normalize_adj_simpgcn(perturbed_graph.adj_matrix, perturbed_graph.node_attr, device=args.dr_device)
            output = eva_model.forward(fea, adj_norm)
        eva_perturbed_label = output.max(1)[1].cpu().numpy()[target]

    # poisoning
    if is_poi:
        poi_model, _ = get_dr_model(name, args, perturbed_graph)
        poi_model.eval()
        poi_perturbed_label = poi_model.output.max(1)[1].cpu().numpy()[target]

    eva_asr = true_label != eva_perturbed_label if eva_perturbed_label is not None else False
    poi_asr = true_label != poi_perturbed_label if poi_perturbed_label is not None else False

    return eva_asr, poi_asr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-m", "--model", default="GCN", type=str, help="model")
    parser.add_argument("--dataset", default="cora", type=str, help="dataset")
    parser.add_argument("--is_gf", default="true", type=str, help="graphgallery / deeprobust")
    parser.add_argument("--times", default=1, type=int)
    parser.add_argument("-irs", "--is_random_seed", default="false", type=str)
    parser.add_argument("-t", "--timestamp", default="", type=str)
    parser.add_argument("-ie", "--is_evasion", default="true", type=str)
    parser.add_argument("-ip", "--is_poisoning", default="true", type=str)
    args = parser.parse_args()
    assert args.timestamp != "", "timestamp is invalid"
    print_args(args)

    args.device = args.device if args.device in ["gpu", "cuda:0", "cuda:1"] and torch.cuda.is_available() else "cpu"
    gg.set_backend("th")

    models = get_models(args.model)
    datasets = get_datasets(args.dataset)

    edgesdir = "result/test_edges/"

    print(datasets)
    print(models)
    for dataset in datasets:
        args.dataset = dataset
        data = NPZDataset(args.dataset,
                          root="~/GraphData/datasets/",
                          verbose=False,
                          transform="standardize")
        graph = data.graph
        splits = data.split_nodes(random_state=15)
        args.splits = splits
        # todo: random times experiments
        if args.is_random_seed == "true":

            continue

        # fixed seed
        is_eva = True if args.is_evasion == "true" else False
        is_poi = True if args.is_poisoning == "true" else False
        results = []
        times = args.times
        for i in range(times):
            cur_result = pd.DataFrame(columns=['eva_asr', 'poi_asr'])
            filename = edgesdir + "_".join([args.dataset, args.timestamp, str(i)]) + '.npy'
            cur_edges = load_json(filename)
            seed = cur_edges['seed']
            del cur_edges['seed']
            embed_types = cur_edges.keys()

            for model_name in models:
                for embed_type in embed_types:
                    targets_edge_flips = cur_edges[embed_type]
                    targets = list(targets_edge_flips.keys())
                    eva_asr = np.zeros(len(targets)).astype('bool')
                    poi_asr = np.zeros(len(targets)).astype('bool')
                    for ti, target in enumerate(targets):
                        edge_flips = targets_edge_flips[target]
                        perturbed_graph = get_perturbed_graph(graph, edge_flips)
                        gf.random_seed(seed, gg.backend())
                        if args.is_gf == "true" or args.dataset == 'ogbn-arxiv':
                            eva_asr[ti], poi_asr[ti] = get_gf_results(model_name, graph, perturbed_graph, args, target, is_eva, is_poi)
                        else:
                            eva_asr[ti], poi_asr[ti] = get_dr_results(model_name, graph, perturbed_graph, args, target, is_eva, is_poi)
                    key = "_".join([embed_type, model_name])
                    cur_result.loc[key] = [eva_asr.mean(), poi_asr.mean()]
            results.append(cur_result)
        save_results(times, results, filename)
