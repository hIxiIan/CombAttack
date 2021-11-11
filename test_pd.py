import argparse
import gc
import random
import graphgallery as gg
from graphgallery.datasets import NPZDataset
from time import strftime, localtime
from ca import run
import os
from utils import save_test
import pandas as pd

DATASETS = ['blockchain30000', 'blockchain40000', 'blockchain50000']

EMBED_TYPE = ['MLP', 'GCN', 'SGC', 'PPNP', 'APPNP', 'SimPGCN',
              'GCN_E']


def get_datasets(cmd_d):
    if len(cmd_d) <= 0:
        return DATASETS
    return cmd_d.split(',')


def get_embed_types(cmd_e):
    if len(cmd_e) <= 0:
        return EMBED_TYPE
    return cmd_e.split(',')


if __name__ == '__main__':
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"], help="code environment")

    parser.add_argument("-st", "--subgraph_type", default="dw_wl", type=str, help="sample method")
    parser.add_argument("-sr", "--sample_ratio", default=0.05, type=float, help="ratio of sampled nodes")
    parser.add_argument("-in_da", "--indirect_attack", action="store_true", help="indirect attack")
    parser.add_argument("-tn", "--target_nums", default=50, type=int, help="target nums")

    parser.add_argument("--dataset", default="", type=str, help="dataset")
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    parser.add_argument("-p", default=7.0, type=float)
    parser.add_argument("-q", default=0.25, type=float)
    parser.add_argument("-a", "--alpha", default=0.25, type=float)
    parser.add_argument("-et", "--embed_type", default="", type=str)
    parser.add_argument('-ip', '--is_phi', default="true", type=str)

    cmd = parser.parse_args()
    gg.set_backend("th")

    rootdir = "result/test_pd"
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    personal = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    dataset_ = get_datasets(cmd.dataset)
    print(dataset_)
    embed_types_ = get_embed_types(cmd.embed_type)
    print(embed_types_)
    subgraph_type = "cluster"

    for dataset in dataset_:
        prefix = "_".join([dataset, personal])
        _prefix = rootdir + os.sep + prefix
        cmd.dataset = dataset

        times = 1
        seeds = [56, 44, 94, 666, 69, 996, 556, 1971, 653, 5018, 2413, 97, 2012, 1997, 21, 32]
        for i in range(times):
            res = pd.DataFrame(columns=['eva_asr', 'eva_asr_wl', 'poi_asr', 'poi_asr_wl', 'cost'])
            cmd.seed = seeds[i]
            filename = "_".join([_prefix, str(i)]) + '.csv'
            print(filename)
            # sga
            try:
                res.loc['sga'] = run("sga", cmd=cmd, verbose=False)
                print('-------sga')
                res.to_csv(filename)
            except Exception as e:
                res.loc['sga'] = [-1, -1, -1, -1, -1]
                print('##################################error', repr(e))

            for embed_type in embed_types_:
                cmd.embed_type = embed_type
                key = '_'.join([embed_type])
                try:
                    res.loc[key] = run(subgraph_type, cmd=cmd, verbose=False)
                except Exception as e:
                    res.loc[key] = [-1, -1, -1, -1, -1]
                    print('##################################error', repr(e))
                print('-------embed_type:{}'.format(embed_type))
                res.to_csv(filename)

        print(_prefix)
        save_test(_prefix, times)
