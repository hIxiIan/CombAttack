import os
import graphgallery as gg
import pandas as pd
import argparse
import warnings
import numpy as np

from time import strftime, localtime
from utils import save_test, get_datasets, get_embed_types, get_mix_types, get_attacked_types, RES_COLUMNS, RES_ERRORS, get_split_atked_types
from ca import run
from test_tedge import get_splits
from tqdm import tqdm
warnings.filterwarnings("ignore")


def do_run(res, key, st, cmd, asr_filename, edges_filename, edges_dict, prefix="", test_assert=False):
    print()
    print('====={}: {} start====='.format(prefix, key))
    if test_assert:
        perturbed_edges_dict, rsps = run(st, cmd=cmd, verbose=False)
        if cmd.edge_flips == "true" and key not in edges_dict:
            edges_dict[key] = perturbed_edges_dict
            np.save(edges_filename + '.npy', edges_dict)
        if cmd.edge_flips == "false":
            for rsp in rsps:
                res.loc['_'.join([key, rsp[0]])] = rsp[1:]
            res.to_csv(asr_filename + '.csv')
    else:
        try:
            perturbed_edges_dict, rsps = run(st, cmd=cmd, verbose=False)
            if cmd.edge_flips == "true" and key not in edges_dict:
                edges_dict[key] = perturbed_edges_dict
                np.save(edges_filename + '.npy', edges_dict)
            if cmd.edge_flips == "false":
                for rsp in rsps:
                    res.loc['_'.join([key, rsp[0]])] = rsp[1:]
                res.to_csv(asr_filename + '.csv')

        except Exception as e:
            res.loc[key] = RES_ERRORS[1:]
            print('=====ASSERT_ERROR:'.format(repr(e)))
    print('====={}: {}   end====='.format(prefix, key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")

    parser.add_argument("-st", "--subgraph_type", default="cluster", type=str, help="sample method")
    parser.add_argument("-sr", "--sample_ratio", default=0.05, type=float, help="ratio of sampled nodes")
    parser.add_argument("-da", "--direct_attack", default="true", type=str, help="direct attack")
    parser.add_argument("-tn", "--target_nums", default=100, type=int, help="target nums")

    parser.add_argument("--dataset", default="", type=str, help="dataset")
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    parser.add_argument("-p", default=7.0, type=float)
    parser.add_argument("-q", default=0.25, type=float)
    parser.add_argument("-a", "--alpha", default=0.25, type=float)
    parser.add_argument("-et", "--embed_type", default="", type=str)
    parser.add_argument('-ip', '--is_phi', default="true", type=str)
    parser.add_argument('-atk', '--atk_model_type', default="", type=str)

    parser.add_argument('--max_iter', default=300, type=int)
    parser.add_argument('--n_init', default=40, type=int)
    parser.add_argument('--times', default=1, type=int)
    parser.add_argument('-tc', '--topk_cluster', default=3, type=int)
    parser.add_argument('-r', '--random', default="false", type=str)
    parser.add_argument('--run_sga', default="true", type=str)
    parser.add_argument('--run_us', default="true", type=str)
    parser.add_argument('-la', '--lay_act', default="layer", type=str)
    parser.add_argument('-lac', '--lay_act_cnt', default=999, type=int)
    parser.add_argument('-dt', '--distance_type', default="euclidean", type=str)
    parser.add_argument('-ns', '--not_split', default="true", type=str)
    parser.add_argument('-tk', '--is_topk', default="false", type=str)
    parser.add_argument('-ef', '--edge_flips', default="false", type=str)
    parser.add_argument('-tm', '--target_mode', default="sur_labels", type=str)
    parser.add_argument('-mc', '--mix_cluster', default="false", type=str)
    parser.add_argument('-mts', '--mix_types', default="MLP,SGC2", type=str)
    parser.add_argument("--test_mode", default="-1", type=str)
    parser.add_argument("--deg_limit", default=2, type=int)
    parser.add_argument("--sur_label_pro_limit", default=0.9, type=float)
    parser.add_argument("--tedge_type", default="", type=str)
    parser.add_argument("--feature_timestamp", default="", type=str)
    parser.add_argument("--features_file", default="", type=str)
    parser.add_argument("--train_size", default=0.5, type=float)
    parser.add_argument("--trans2vec_model", default="ocsvm", type=str)
    parser.add_argument('--bmbc_mode', default="false", type=str)
    cmd = parser.parse_args()
    cmd.hids = None
    cmd.acts = None
    cmd.weight_decay = None
    cmd.lr = None
    gg.set_backend("th")

    edgesdir = "result/test_edges"
    rootdir = "result/test_cluster"
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    if not os.path.exists(edgesdir):
        os.mkdir(edgesdir)
    curtime = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    dataset_ = get_datasets(cmd.dataset)
    embed_types_ = get_embed_types(cmd.embed_type)
    atked_types_ = get_attacked_types(cmd.atk_model_type)
    tedge_types_ = get_splits(cmd.tedge_type)
    not_split = True if cmd.not_split == "true" else False
    mix_cluster = True if cmd.mix_cluster == "true" else False
    mix_types_ = get_mix_types(cmd.mix_types)
    method_types = embed_types_
    if mix_cluster:
        method_types = mix_types_
    print(dataset_)
    print(atked_types_)
    print(embed_types_)
    print(mix_types_)
    print(tedge_types_)

    for dataset in dataset_:
        if dataset in ["tedge", "trans2vec"]:
            assert len(cmd.feature_timestamp) > 0, 'feature_timestamp error'
            cmd.edge_flips = "true"
        if "bc" in dataset and cmd.bmbc_mode == "true":
            cmd.edge_flips = "true"
        cmd.dataset = dataset
        _asr_prefix = rootdir + os.sep + "_".join([cmd.dataset, curtime])
        _edges_prefix = edgesdir + os.sep + "_".join([cmd.dataset, curtime])
        seeds = [2022, 2012, 1997, 5018, 2413, 97, 21, 32, 56, 44, 94]
        # seeds = [2012, 1997, 5018, 2413, 2022, 97, 21, 32, 56, 44, 94]
        times = min(cmd.times, len(seeds))
        for i in range(times):
            edge_dict = {}
            res = pd.DataFrame(columns=RES_COLUMNS[1:])
            cmd.seed = seeds[i]
            asr_filename = "_".join([_asr_prefix, str(i)])
            edges_filename = "_".join([_edges_prefix, str(i)])
            edge_dict['seed'] = cmd.seed
            print(cmd.seed)
            print(asr_filename)
            print(edges_filename)

            count = 0
            total = len(tedge_types_)
            if cmd.run_sga == "true":
                if dataset in ["tedge", "trans2vec"]:
                    cmd.atk_model_type = ','.join(atked_types_)
                    for tedge_type in tedge_types_:
                        count += 1
                        cmd.features_file = "_".join([tedge_type, cmd.feature_timestamp, str(i)])
                        print('\ndataset: {}, times:{}, {}/{}; sga: {} attack atked_type: {}, tedge_type: {}'.format(
                            dataset, i, count, total, "sga", cmd.atk_model_type, tedge_type))
                        key = '_'.join(['sga', tedge_type])
                        do_run(res, key, 'sga', cmd, asr_filename, edges_filename, edge_dict, key)
                else:
                    cmd.atk_model_type = ','.join(atked_types_)
                    count += 1
                    print('\ndataset: {}, times:{}, {}/{}; sga: {} attack atked_type: {}'.format(
                        dataset, i, count, total, "sga", cmd.atk_model_type))
                    key = '_'.join(['sga'])
                    do_run(res, key, 'sga', cmd, asr_filename, edges_filename, edge_dict, key)
            if cmd.run_us != "true":
                continue

            count = 0
            total = 0
            for embed_type in method_types:
                cur_count = len(get_split_atked_types(dataset, embed_type, atked_types_, not_split))
                if dataset in ["tedge", "trans2vec"]:
                    cur_count *= len(tedge_types_)
                total += cur_count

            for embed_type in method_types:
                cmd.embed_type = embed_type if not mix_cluster else None
                split_atked_types_ = get_split_atked_types(dataset, embed_type, atked_types_, not_split)
                for atked_type in split_atked_types_:
                    if dataset in ["tedge", "trans2vec"]:
                        cmd.atk_model_type = ','.join(atked_type)
                        for tedge_type in tedge_types_:
                            count += 1
                            cmd.features_file = "_".join([tedge_type, cmd.feature_timestamp, str(i)])
                            print('\ndataset: {}, times:{}, {}/{}; embed_type: {} attack atked_type: {}, tedge_type: {}'.format(dataset, i, count, total, embed_type, cmd.atk_model_type, tedge_type))
                            key = '_'.join(['&'.join(embed_type), tedge_type]) if mix_cluster else '_'.join([embed_type, tedge_type])
                            prefix = '_'.join(['&'.join(embed_type), tedge_type]) if mix_cluster else '_'.join([embed_type, tedge_type])
                            do_run(res, key, 'cluster', cmd, asr_filename, edges_filename, edge_dict, prefix)
                    else:
                        count += 1
                        cmd.atk_model_type = ','.join(atked_type)
                        print('\ndataset: {}, times:{}, {}/{}; embed_type: {} attack atked_type: {}'.format(dataset, i, count, total, embed_type, cmd.atk_model_type))
                        key = '&'.join(embed_type) if mix_cluster else '_'.join([embed_type])
                        do_run(res, key, 'cluster', cmd, asr_filename, edges_filename, edge_dict, embed_type)
            print('dataset:{}, times:{}'.format(dataset, i))
        print(_asr_prefix)
        print(_edges_prefix)
        if cmd.edge_flips == "false":
            save_test(_asr_prefix, times, seeds[:times])