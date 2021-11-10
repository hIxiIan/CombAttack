import argparse
import gc
import random
import graphgallery as gg
from graphgallery.datasets import NPZDataset
from ca import run


if __name__ == '__main__':
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"], help="code environment")

    parser.add_argument("-st", "--subgraph_type", default="dw_wl", type=str, help="sample method")
    parser.add_argument("-sr", "--sample_ratio", default=0.05, type=float, help="ratio of sampled nodes")
    parser.add_argument("-in_da", "--indirect_attack", action="store_true", help="indirect attack")

    parser.add_argument("--dataset", default="blockchain30000", type=str, help="dataset")
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    parser.add_argument("-p", default=7.0, type=float)
    parser.add_argument("-q", default=0.25, type=float)
    parser.add_argument("-a", "--alpha", default=0.25, type=float)
    parser.add_argument("-et", "--embed_type", default="MLP", type=str)

    cmd = parser.parse_args()
    random.seed(cmd.seed)
    gg.set_backend("th")
    data = NPZDataset(cmd.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False)

    graph = data.graph
    # splits = data.split_nodes(train=1.0, test=0.0, val=0.0, random_state=15)
    train_nodes = [i for i in range(graph.adj_matrix.shape[0])]
    surrogate_model = gg.gallery.nodeclas.SGCPD(device="cpu", seed=1000).setup_graph(graph, K=1).build()

    his = surrogate_model.fit(train_nodes,
                              verbose=2,
                              epochs=6)
    gc.collect()
    print('success')
