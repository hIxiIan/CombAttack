import os
from time import strftime, localtime

import graphgallery as gg
import pandas as pd
import argparse

from utils import save_test
from ca import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"], help="code environment")

    parser.add_argument("-st", "--subgraph_type", default="dw_wl", type=str, help="sample method")
    parser.add_argument("-sr", "--sample_ratio", default=0.05, type=float, help="ratio of sampled nodes")
    parser.add_argument("-in_da", "--indirect_attack", action="store_true", help="indirect attack")
    parser.add_argument("-tn", "--target_nums", default=1000, type=int, help="target nums")

    parser.add_argument("--dataset", default="cora", type=str, help="dataset")
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    parser.add_argument("-p", default=7.0, type=float)
    parser.add_argument("-q", default=0.25, type=float)
    parser.add_argument("-a", "--alpha", default=0.25, type=float)
    parser.add_argument("-et", "--embed_type", default="MLP", type=str)

    cmd = parser.parse_args()
    gg.set_backend("th")

    rootdir = "result/test_dw"
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    personal = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    prefix = "_".join([cmd.dataset, personal])
    _prefix = rootdir + os.sep + prefix

    times = 1
    seeds = [2012, 1997, 5018, 2413, 97, 21, 32, 56, 44, 94]
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
            res.loc['sga'] = [-1, -1, -1]
            print('##################################error', repr(e))

        # us
        subgraph_types = ['dw', 'dw_biased',
                          'dw_wl', 'dw_wl_dynamic', 'dw_wl_gains', 'dw_wl_kh', 'dw_biased_wl',
                          'dw_ce', 'dw_ce_dynamic',
                          'dw_purity', 'dw_purity_gains', 'dw_purity_gains_select', 'dw_biased_purity',
                          'n2v', 'n2v_wl', 'n2v_ce', 'n2v_purity']
        for subgraph_type in subgraph_types:
            p = 1.0
            q = 1.0
            key = '_'.join([subgraph_type, str(p), str(q)])
            try:
                res.loc[key] = run(subgraph_type, cmd=cmd, p=p, q=q, verbose=False)
            except Exception as e:
                res.loc[key] = [-1, -1, -1]
                print('##################################error', repr(e))
            print('-------subgraph:{}, p={}, q={}'.format(subgraph_type, p, q))
            res.to_csv(filename)

        # subgraph_types = ['dw_biased']
        # # 'dw_ce', 'dw_ce_topk'
        # for subgraph_type in subgraph_types:
        #     for p in [1.5, 2.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 12.0, 14.0, 16.0, 18.0]:
        #         for q in [0.1, 0.25, 0.4, 0.5, 0.6, 0.8]:
        #             key = '_'.join([subgraph_type, str(p), str(q)])
        #             try:
        #                 res.loc[key] = run(subgraph_type, cmd=cmd, p=p, q=q, verbose=False)
        #             except Exception as e:
        #                 res.loc[key] = [-1, -1, -1]
        #                 print('##################################error', repr(e))
        #             print('-------subgraph:{}, p={}, q={}'.format(subgraph_type, p, q))
        #             res.to_csv(filename)
    print(_prefix)
    save_test(_prefix, times)



