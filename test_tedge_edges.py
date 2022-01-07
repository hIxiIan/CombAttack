import argparse
import random
import pandas as pd
import numpy as np
import torch
from time import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tedge import load_labels, tGraph, tGraphNE, METHOD_MAP, random_seed
from copy import deepcopy as dc


ds = ["SVM"]


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


def get_balance(g, u):
    u_transfer_out = np.sum([tu[2] for tu in g.edges(u, data='weight')])
    transfer_in = []
    for pre_node in g.predecessors(u):
        for tu in g.edges(pre_node, data='weight'):
            if tu[1] == u:
                transfer_in.append(tu[2])
    transfer_in = np.sum(transfer_in)
    u_transfer_in = transfer_in
    u_balance = u_transfer_in - u_transfer_out
    return u_balance


def get_perturbed_graph(tG_ori, edge_flips):
    tG = dc(tG_ori)
    adj_flips = get_adj_flips(edge_flips)
    if adj_flips is not None:
        g = tG.G
        for edge in adj_flips:
            u, v = str(edge[0]), str(edge[1])
            if g.has_edge(u, v):
                timestamps = list(g[u][v].keys())
                timestamp = timestamps[-1] + int(np.diff(timestamps).mean())
                amount = np.mean([g[u][v][timestamp]['weight'] for timestamp in timestamps])
                if g.has_edge(u, v, timestamp):
                    if g[u][v][timestamp]['weight'] != amount:
                        g[u][v][timestamp]['weight'] += amount
                else:
                    g.add_edge(u, v, key=timestamp, weight=amount)
            else:
                timestamp = random.randint(tG.max_time, tG.max_time + 500000)
                his_amount = [np.mean([weight['weight'] for weight in list(nbr.values())]) for nbr in list(g[u].values())]
                if len(his_amount) > 0:
                    amount = np.quantile(his_amount, 0.5)
                else:
                    u_balance = get_balance(g, u)
                    amount = random.uniform(0, u_balance)
                g.add_edge(u, v, key=timestamp, weight=amount)
        tG.G = g
    return tG


def get_sklearn_results(eva_model, name, true_label, perturbed_tG, args, target, tedge_type, is_eva, is_poi):
    eva_perturbed_label = None
    poi_perturbed_label = None

    time_biased_type, first_biased_type, amount_biased, alpha = METHOD_MAP[tedge_type]
    tGNE = tGraphNE(perturbed_tG, time_biased_type, first_biased_type, amount_biased, alpha,
                   seed=args.seed, verbose=args.verbose, output="", is_test_tedge_edges=True, is_dan=args.is_dan)
    perturbed_features = tGNE.features

    # evasion
    if is_eva:
        eva_perturbed_label = eva_model.predict([perturbed_features[args.nodes_to_keep][target]])[0]

    # poisoning
    if is_poi:
        poi_model = get_sklean_model(name, args)
        nodes_embeddings = pd.DataFrame(perturbed_features[args.nodes], index=args.nodes)
        X_train, X_test, y_train, y_test = train_test_split(nodes_embeddings, args.nodes_labels,
                                                            train_size=args.train_size,
                                                            random_state=args.seed)
        poi_model.fit(X_train, y_train)
        poi_perturbed_label = poi_model.predict([perturbed_features[args.nodes_to_keep][target]])[0]

    eva_asr = true_label != eva_perturbed_label if eva_perturbed_label is not None else False
    poi_asr = true_label != poi_perturbed_label if poi_perturbed_label is not None else False

    return eva_asr, poi_asr


def get_true_labels(attacked_models, args, tedgedir, i):
    tedge_types = list(attacked_models.keys())
    true_labels = {}
    for model_name_tedge_type in tedge_types:
        model_name, tedge_type = model_name_tedge_type.split('_')
        attacked_model = attacked_models[model_name + "_" + tedge_type]
        tedge_features_file = tedgedir + "_".join([tedge_type, args.tedge_timestamp, str(i)]) + '.csv'
        embeddings = pd.read_csv(tedge_features_file).values
        # nodes_embeddings = pd.DataFrame(embeddings[args.nodes], index=args.nodes)
        true_labels['_'.join([model_name, tedge_type])] = attacked_model.predict(embeddings[args.nodes_to_keep])
    return true_labels


def get_sklean_model(model_name, args):
    if model_name == "SVM":
        model = SVC(kernel='linear', C=0.4, random_state=args.seed)
        model.name = model_name
        return model

    assert False, "get_sklean_model invalid"


def get_attacked_models(models, args, tedgedir, i, embed_types_tedge_types):
    tedge_types = list(set([key.split('_')[-1] for key in embed_types_tedge_types]) - {'seed'})
    attacked_models = {}
    attacked_models_acc = {}
    for model_name in models:
        for tedge_type in tedge_types:
            tedge_features_file = tedgedir + "_".join([tedge_type, args.tedge_timestamp, str(i)]) + '.csv'
            embeddings = pd.read_csv(tedge_features_file).values
            nodes_embeddings = pd.DataFrame(embeddings[args.nodes], index=args.nodes)
            model = get_sklean_model(model_name, args)
            X_train, X_test, y_train, y_test = train_test_split(nodes_embeddings, args.nodes_labels,
                                                                train_size=args.train_size,
                                                                random_state=args.seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = (y_pred == y_test).mean()
            print('atked_model: {}, tedge_type: {}, clean_acc: {}'.format(model_name, tedge_type, acc))

            key = '_'.join([model_name, tedge_type])
            attacked_models[key] = model
            attacked_models_acc[key] = acc
    args.attacked_models_acc = attacked_models_acc
    return attacked_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-m", "--model", default="", type=str, help="model")
    parser.add_argument("--dataset", default="tedge", type=str, help="dataset")
    parser.add_argument("--times", default=1, type=int)
    parser.add_argument("-tt", "--tedge_timestamp", default="2022_01_04_23_29_22", type=str)
    parser.add_argument("-eft", "--edge_flips_timestamp", default="2022_01_05_18_09_43", type=str)
    parser.add_argument("--train_size", default=0.5, type=float)
    parser.add_argument("-ie", "--is_evasion", default="true", type=str)
    parser.add_argument("-ip", "--is_poisoning", default="true", type=str)
    parser.add_argument('--run_sga', default="true", type=str)
    parser.add_argument('--run_us', default="true", type=str)
    parser.add_argument('--is_dan_mode', default="true", type=str)
    args = parser.parse_args()

    # args.tedge_timestamp = "2022_01_04_23_29_22"
    # args.edge_flips_timestamp = "2022_01_05_18_09_43"
    assert args.tedge_timestamp != "", "tedge_timestamp is invalid"
    assert args.edge_flips_timestamp != "", "edge_flips_timestamp is invalid"
    print_args(args)

    args.device = args.device if args.device in ["gpu", "cuda:0", "cuda:1"] and torch.cuda.is_available() else "cpu"

    models = get_models(args.model)
    datasets = [args.dataset]

    edgesdir = "result/test_edges/"
    tedgedir = 'result/test_tedge/'

    print(datasets)
    print(models)
    count = 0
    total = args.times * len(datasets)
    for dataset in datasets:
        tG_ori = tGraph('dataset/phishing/TransEdgelist.txt', verbose=args.verbose)
        args.dataset = dataset
        # fixed seed
        is_eva = True if args.is_evasion == "true" else False
        is_poi = True if args.is_poisoning == "true" else False
        args.is_dan = True if args.is_dan_mode == "true" else False
        results = []
        times = args.times

        for i in range(times):
            sample_labels = load_labels('dataset/phishing/label.txt')
            args.nodes = list([int(node) for node in sample_labels.keys()])
            args.nodes_labels = list(sample_labels.values())
            args.nodes_to_keep = pd.read_csv('dataset/phishing/tedge_nodes_to_keep.csv').values.ravel()
            count += 1
            print('\ncount:{}/{}, dataset:{}, times:{}/{}'.format(count, total, args.dataset, i + 1, times))
            cur_result = pd.DataFrame(columns=['eva_asr', 'poi_asr', 'clean_acc'])
            filename = edgesdir + "_".join([args.dataset, args.edge_flips_timestamp, str(i)])
            cur_edges = load_json(filename)
            seed = cur_edges['seed']
            args.seed = seed
            del cur_edges['seed']
            embed_types_tedge_types = cur_edges.keys()

            attacked_models = get_attacked_models(models, args, tedgedir, i, embed_types_tedge_types)
            true_labels = get_true_labels(attacked_models, args, tedgedir, i)
            eva_asr = {}
            poi_asr = {}
            for embed_type_tedge_type in embed_types_tedge_types:
                embed_type, tedge_type = embed_type_tedge_type.split('_')
                if args.run_sga == "false" and "sga" in embed_type.lower():
                    print('skip embed_type_tedge_type:{}'.format(embed_type_tedge_type))
                    continue
                if args.run_us == "false" and "sga" not in embed_type.lower():
                    print('skip embed_type_tedge_type:{}'.format(embed_type_tedge_type))
                    continue
                args.tedge_features_file = tedgedir + "_".join([tedge_type, args.tedge_timestamp, str(i)]) + '.csv'

                start = time()
                print('embed_type:{}, tedge_type: {}, atk_models:{}'.format(embed_type, tedge_type, models))
                targets_edge_flips = cur_edges[embed_type_tedge_type]
                targets = list(targets_edge_flips.keys())
                # !!!!! targets all in 445 phishing nodes normally
                random_seed(seed)
                for ti, target in enumerate(targets):
                    t1 = time()
                    print('attack target: {}. {}/{} '.format(target, ti + 1, len(targets)))
                    edge_flips = targets_edge_flips[target]
                    perturbed_tG = get_perturbed_graph(tG_ori, edge_flips)

                    for model_name in models:
                        _key = "_".join([model_name, tedge_type])
                        attacked_model = attacked_models[_key]
                        true_label = true_labels[_key][target]
                        t1 = time()
                        is_eva_success, is_poi_success = get_sklearn_results(attacked_model, model_name,
                                                                             true_label, perturbed_tG,
                                                                             args, target, tedge_type, is_eva, is_poi)
                        print('get_sklearn_results cost: {} min'.format((time() - t1) / 60))
                        key = "_".join([embed_type, _key])
                        if key not in eva_asr:
                            eva_asr[key] = np.zeros(len(targets)).astype('bool')
                        if key not in poi_asr:
                            poi_asr[key] = np.zeros(len(targets)).astype('bool')
                        eva_asr[key][ti] = is_eva_success
                        poi_asr[key][ti] = is_poi_success
                    print('attack target: {}. cost: {} min'.format(target, (time() - t1) / 60))

                for model_name in models:
                    _key = "_".join([model_name, tedge_type])
                    key = "_".join([embed_type, _key])
                    cur_result.loc[key] = [eva_asr[key].mean(), poi_asr[key].mean(), args.attacked_models_acc[_key]]
                cur_result.to_csv(filename + '.csv')
                print('cost:{} min'.format((time() - start) / 60))
            results.append(cur_result)
        save_results(times, results, filename)
