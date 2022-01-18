import argparse
import os
import pandas as pd
import numpy as np
import torch
import graphgallery as gg
import graphgallery.functional as gf
import scipy.sparse as sp
from ca import get_model, get_dr_model, get_attacked_models
from graphgallery.datasets import NPZDataset
from time import strftime, localtime
from utils import load_pickle
from time import time
from bm_gcn import get_output
from copy import deepcopy as dc

MODELS = ["BMGCN"]
DATASETS = ["bc1"]


def load_json(filename):
    if 'npy' not in filename:
        filename += '.npy'
    return np.load(filename, allow_pickle=True).item()


def get_models(models):
    if len(models) <= 0:
        return MODELS
    return models.split(",")


def get_datasets(datasets):
    if len(datasets) <= 0:
        return DATASETS
    return datasets.split(",")


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


# todo mul_dG正常来说需要更新插入的连边，但考虑到攻击时不会重复攻击，所以可以不处理
def get_perturbed_tuple(ori_data, edge_flips):
    A, V, F, mul_dG = ori_data[0].tolil(copy=True), ori_data[1].tolil(copy=True), ori_data[2].tolil(copy=True), ori_data[3]
    adj_flips = get_adj_flips(edge_flips)

    if adj_flips is not None:
        for edge in adj_flips:
            u, v = edge[0], edge[1]
            # not exist edges original
            if A[u, v] == 0 and V[u, v] == 0:
                A[u, v] = A[u, v] + 1.
                V[u, v] = 0.
                F[u, v] = 0.
            else:
                # exist edges original
                _d = list(mul_dG[u][v].values())
                _timestamp = [_dd['timestamp'] for _dd in _d]
                next_timestamp = np.random.randint(_timestamp[-1], _timestamp[-1] + 500000)
                _timestamp.append(next_timestamp)
                V[u, v] = np.var(_timestamp)
                A[u, v] = len(_timestamp)
                if A[u, v] >= 2.:
                    F[u, v] = np.diff(_timestamp).sum() / A[u, v]

    return A.tocsr(), V.tocsr(), F.tocsr()


def get_gf_results(eva_model, name, true_label, perturbed_tuple, args, target, is_eva, is_poi):
    eva_perturbed_label = None
    poi_perturbed_label = None
    A, V, F = perturbed_tuple[0], perturbed_tuple[1], perturbed_tuple[2]
    # evasion
    if is_eva:
        eva_perturbed_label = get_output(eva_model, [A, V, F])[target].argmax()

    # poisoning
    if is_poi:
        poi_model = get_model(name, args, [A, V, F])
        poi_perturbed_label = get_output(poi_model)[target].argmax()

    eva_asr = true_label != eva_perturbed_label if eva_perturbed_label is not None else False
    poi_asr = true_label != poi_perturbed_label if poi_perturbed_label is not None else False

    return eva_asr, poi_asr


def get_true_labels(attacked_models):
    true_labels = []
    for attacked_model in attacked_models:
        assert attacked_model[0].name == "BMGCN", "get_true_labels invalid"
        output = get_output(attacked_model)
        true_label = [logit.argmax() for logit in output]
        true_labels.append(true_label)
    return true_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-m", "--model", default="", type=str, help="model")
    parser.add_argument("--dataset", default="", type=str, help="dataset")
    parser.add_argument("--times", default=1, type=int)
    parser.add_argument("-t", "--timestamp", default="", type=str)
    parser.add_argument("-ie", "--is_evasion", default="true", type=str)
    parser.add_argument("-ip", "--is_poisoning", default="true", type=str)
    parser.add_argument('--run_sga', default="true", type=str)
    parser.add_argument('--run_us', default="true", type=str)
    parser.add_argument('-tm', '--target_mode', default="sur_labels", type=str)
    args = parser.parse_args()
    # args.timestamp = "2021_12_24_13_56_01"
    assert args.timestamp != "", "timestamp is invalid"
    print_args(args)

    args.device = args.device if args.device in ["gpu", "cuda:0", "cuda:1"] and torch.cuda.is_available() else "cpu"
    gg.set_backend("th")

    models = get_models(args.model)
    datasets = get_datasets(args.dataset)

    edgesdir = "result/test_edges/"

    print(datasets)
    print(models)
    count = 0
    total = args.times * len(datasets)
    for dataset in datasets:
        args.dataset = dataset
        data = NPZDataset(args.dataset,
                          root="~/GraphData/datasets/",
                          verbose=False,
                          transform="standardize")
        graph = data.graph
        splits = data.split_nodes(random_state=15)
        args.splits = splits
        SAMPLE_GSIZE = int(args.dataset.split('bc')[-1]) * 10000
        SAMPLE_MULDIGS_PATH = os.path.join(data.root, 'graph_%d/SP_MulDiGs.pkl' % SAMPLE_GSIZE)
        args.DATA_PATH = os.path.join(data.root, 'graph_%d' % SAMPLE_GSIZE)
        mul_dG = load_pickle(SAMPLE_MULDIGS_PATH)
        bmbc = np.load(''.join([data.root, os.sep, "bm" + args.dataset + ".npz"]), allow_pickle=True)
        A = bmbc["A"].item()
        V = bmbc["V"].item()
        F = bmbc["F"].item()
        ori_data = tuple(A, V, F, mul_dG)

        # fixed seed
        is_eva = True if args.is_evasion == "true" else False
        is_poi = True if args.is_poisoning == "true" else False
        results = []
        times = args.times

        for i in range(times):
            count += 1
            print('\ncount:{}/{}, dataset:{}, times:{}/{}'.format(count, total, args.dataset, i + 1, times))
            cur_result = pd.DataFrame(columns=['eva_asr', 'poi_asr', 'clean_acc'])
            filename = edgesdir + "_".join([args.dataset, args.timestamp, str(i)])
            cur_edges = load_json(filename)
            seed = cur_edges['seed']
            del cur_edges['seed']
            embed_types = cur_edges.keys()

            attacked_models = get_attacked_models(models, args, graph)
            true_labels = get_true_labels(attacked_models)
            eva_asr = {}
            poi_asr = {}
            for embed_type in embed_types:
                if args.run_sga == "false" and "sga" in embed_type.lower():
                    print('skip embed_type:{}'.format(embed_type))
                    continue
                if args.run_us == "false" and "sga" not in embed_type.lower():
                    print('skip embed_type:{}'.format(embed_type))
                    continue
                start = time()
                print('embed_type:{}, atk_models:{}'.format(embed_type, models))
                targets_edge_flips = cur_edges[embed_type]
                targets = list(targets_edge_flips.keys())
                for ti, target in enumerate(targets):
                    if ti % 20 == 0:
                        print('{} targets attacked'.format(ti))
                    edge_flips = targets_edge_flips[target]
                    perturbed_tuple = get_perturbed_tuple(ori_data, edge_flips)
                    for ai, attacked_model in enumerate(attacked_models):
                        gf.random_seed(seed, gg.backend())
                        name = attacked_model[0].name
                        is_eva_success, is_poi_success = get_gf_results(attacked_model, name, true_labels[ai][target], perturbed_tuple, args, target, is_eva, is_poi)
                        key = "_".join([embed_type, name])
                        if key not in eva_asr:
                            eva_asr[key] = np.zeros(len(targets)).astype('bool')
                        if key not in poi_asr:
                            poi_asr[key] = np.zeros(len(targets)).astype('bool')
                        eva_asr[key][ti] = is_eva_success
                        poi_asr[key][ti] = is_poi_success
                for ai, attacked_model in enumerate(attacked_models):
                    name = attacked_model[0].name
                    key = "_".join([embed_type, name])
                    cur_result.loc[key] = [eva_asr[key].mean(), poi_asr[key].mean(), args.attacked_models_acc[ai]]
                cur_result.to_csv(filename + '.csv')
                print('cost:{} min'.format((time() - start) / 60))
            results.append(cur_result)
        save_results(times, results, filename)
