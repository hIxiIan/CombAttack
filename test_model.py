import argparse
import torch
import graphgallery as gg
import graphgallery.functional as gf
from ca import get_model, get_dr_model
from graphgallery.datasets import NPZDataset

ds = ["GCN", "SGC", "SimPGCN", "GCN_Jaccard", "FastGCN"]


def get_models(models):
    if len(models) <= 0:
        return ds
    return models.split(",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-m", "--model", default="RobustGCN", type=str, help="model")
    parser.add_argument("--dataset", default="cora", type=str, help="dataset")
    parser.add_argument("--is_gf", default="true", type=str, help="graphgallery / deeprobust")
    args = parser.parse_args()
    args.device = args.device if args.device in ["gpu", "cuda:0", "cuda:1"] and torch.cuda.is_available() else "cpu"

    gg.set_backend("th")
    data = NPZDataset(args.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")
    graph = data.graph
    splits = data.split_nodes(random_state=15)
    gf.random_seed(args.seed, gg.backend())

    models = get_models(args.model)
    for model_name in models:
        if args.is_gf == "true":
            model = get_model(model_name, args, graph)
            model.fit(splits.train_nodes, splits.val_nodes, verbose=args.verbose, epochs=200)
            results = model.evaluate(splits.test_nodes, verbose=args.verbose)
            print(f'Model: {model_name}, Test accuracy {results.accuracy:.2%}')
        else:
            model, acc = get_dr_model(model_name, args, graph)
            print(f'Model: {model_name}, Test accuracy {acc:.2%}')