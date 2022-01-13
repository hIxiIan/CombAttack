import argparse
import os
from tedge import run_tedge, random_seed
from time import strftime, localtime


TEDGE_TYPES = ['TEDGE', 'TBS', 'WBS', 'TBS+WBS']


def get_splits(splits, sep=','):
    if len(splits) <= 0:
        return TEDGE_TYPES
    return splits.split(sep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-tt", "--tedge_type", default="", choices=['TEDGE', 'TBS', 'WBS', 'TBS+WBS'], type=str)
    parser.add_argument("--run_emb", default="true", type=str)
    parser.add_argument("--run_nc", default="true", type=str)
    parser.add_argument("-f", "--filename", default="", type=str)
    parser.add_argument("-d", "--dimensions", default=128, type=int) # 128
    parser.add_argument("--num_walks", default=4, type=int) # 4
    parser.add_argument("--walk_length", default=10, type=int) # 10
    parser.add_argument("--window_size", default=4, type=int) # 4
    parser.add_argument("--workers", default=1, type=int) # 8
    parser.add_argument("--train_size", default=0.5, type=float)
    parser.add_argument('--times', default=1, type=int)
    args = parser.parse_args()

    random_seed(args.seed)
    outputdir = "result/test_tedge"
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    args.outputdir = outputdir
    args.curtime = strftime("%Y_%m_%d_%H_%M_%S", localtime())

    tedge_types = get_splits(args.tedge_type)
    print(tedge_types)
    seeds = [2022, 2012, 1997, 5018, 2413, 97, 21, 32, 56, 44, 94]
    times = min(args.times, len(seeds))

    count = 0
    total = times * len(tedge_types)
    for i in range(times):
        args.i = i
        for tedge_type in tedge_types:
            count += 1
            print('times:{}, count: {}/{}, tedge_type:{}'.format(i, count, total, tedge_type))
            args.tedge_type = tedge_type
            run_tedge(args)