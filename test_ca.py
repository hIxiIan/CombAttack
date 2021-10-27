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
    cmd = parser.parse_args()
    gg.set_backend("th")


    rootdir = "result/test_ca"
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    personal = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    prefix = "_".join([cmd.dataset, personal])
    _prefix = rootdir + os.sep + prefix

    times = 3
    seeds = [5018, 2413, 97, 56, 44, 94, 2012, 1997, 21, 32]
    for i in range(times):
        res = pd.DataFrame(columns=['acc', 'wlacc', 'cost'])
        cmd.seed = seeds[i]
        filename = "_".join([_prefix, str(i)]) + '.csv'
        print(filename)
        # sga
        try:
            acc, wlacc, cost = run("sga", cmd=cmd, us=False, verbose=False)
            res.loc['sga'] = [acc, wlacc, cost]
            print('-------sga')
            res.to_csv(filename)
        except Exception as e:
            res.loc['sga'] = [-1, -1, -1]
            print('##################################error', repr(e))

        # us
        subgraph_types = ['dw', 'dw_purity', 'dw_wl', 'dw_kh', 'dw_ce', 'n2v', 'n2v_purity', 'n2v_wl', 'n2v_ce',
                          'spread_random_wl', 'spread_random_wl_kh', 'spread_random_ce',
                          'spread_random_ce_kh', 'spread_ce', 'ppr', 'ppr_wl']

        # dw
        subgraph_types = ['dw',
                          'dw_wl', 'dw_wl_dynamic', 'dw_wl_gains',
                          'dw_purity', 'dw_purity_gains', 'dw_purity_gains_select',
                          'dw_ce', 'dw_ce_dynamic']
        for subgraph_type in subgraph_types:
            p = 1.0
            q = 1.0
            key = '_'.join([subgraph_type, str(p), str(q)])
            try:
                acc, wlacc, cost = run(subgraph_type, cmd=cmd, p=p, q=q, verbose=False)
                res.loc[key] = [acc, wlacc, cost]
            except Exception as e:
                res.loc[key] = [-1, -1, -1]
                print('##################################error', repr(e))
            print('-------subgraph:{}, p={}, q={}'.format(subgraph_type, p, q))
            res.to_csv(filename)

        # dw
        # subgraph_types = ['n2v_wl', 'n2v_purity']
        # for subgraph_type in subgraph_types:
        #     for p in [7.0]:
        #         for q in [0.25]:
        #             key = '_'.join([subgraph_type, str(p), str(q)])
        #             try:
        #                 acc, wlacc, cost = run(subgraph_type, cmd=cmd, p=p, q=q, verbose=False)
        #                 res.loc[key] = [acc, wlacc, cost]
        #             except Exception as e:
        #                 res.loc[key] = [-1, -1, -1]
        #                 print('##################################error', repr(e))
        #             print('-------subgraph:{}, p={}, q={}'.format(subgraph_type, p, q))
        #             res.to_csv(filename)

        # spread
        subgraph_types = ['spread_wl']
        for subgraph_type in subgraph_types:
            key = '_'.join([subgraph_type])
            p = 1.0
            q = 1.0
            try:
                acc, wlacc, cost = run(subgraph_type, cmd=cmd, p=p, q=q, verbose=False)
                res.loc[key] = [acc, wlacc, cost]
            except Exception as e:
                res.loc[key] = [-1, -1, -1]
                print('##################################error', repr(e))
            print('-------subgraph:{}'.format(subgraph_type))
            res.to_csv(filename)

        # ppr
        # 和random seed无关
        subgraph_types = ['ppr_wl_topk_asc']
        for subgraph_type in subgraph_types:
            p = 1.0
            q = 1.0
            for alpha in [0.01]:
                key = '_'.join([subgraph_type, str(alpha)])
                try:
                    acc, wlacc, cost = run(subgraph_type, cmd=cmd, p=p, q=q, alpha=alpha, verbose=False)
                    res.loc[key] = [acc, wlacc, cost]
                except Exception as e:
                    res.loc[key] = [-1, -1, -1]
                    print('##################################error', repr(e))
                print('-------subgraph:{}, alpha={}'.format(subgraph_type, alpha))
                res.to_csv(filename)

    print(_prefix)
    save_test(_prefix, times)
