import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.data import PtbDataset, PrePtbDataset
import argparse
import warnings

warnings.filterwarnings("ignore")

import time
import sys
from ogb.nodeproppred import PygNodePropPredDataset
import copy


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()

def get_dataset(dataset):
    if len(dataset) == 0:
        return DATASETS
    return dataset.split(',')


def get_type(types):
    if len(types) == 0:
        return TYPES
    return types.split(',')


def get_gnns(gnns):
    if len(gnns) == 0:
        return GNNS
    return gnns.split(',')

parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('--type', type=str, default='nma', help='type')
parser.add_argument('--gnns', type=str, default='clustergcn', help='type')
parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')
parser.add_argument('--cal_pre', type=bool, default=False, help='cal pre')
parser.add_argument('--timestamp', type=str, default="")
parser.add_argument('--is_test', type=str, default="true")
parser.add_argument('--is_eva', type=str, default="true")
parser.add_argument('--is_poi', type=str, default="false")
parser.add_argument('--test_nodes_nums', type=int, default=20)
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
DATASETS = ['pubmed', 'citeseer']
TYPES = ['nga', 'nma', 'nmab', 'nettack', 'sga', 'fga']
GNNS = ['gcn', 'sgc', 'gat', 'rgcn', 'jaccard']

begin_time = time.time()
time_local = time.localtime(int(begin_time))
total_time = time.strftime("%Y%m%d_%H%M%S", time_local)
results = pd.DataFrame(columns=['eva_asr', 'poi_asr', 'avg_margin_decrease'])
finished = []
if len(args.timestamp) > 0:
    total_time = args.timestamp
    results = pd.read_csv('kdd_test_' + args.timestamp + '.csv', index_col=0)
    finished = list(results.index)
    print(results)

datasets = get_dataset(args.dataset)
types = get_type(args.type)
gnns = get_gnns(args.gnns)
total = len(datasets) * len(types) * len(gnns)
cnt = 0
print(datasets)
print(types)
print(gnns)
for dataset in datasets:
    args.dataset = dataset
    for _type in types:
        args.type = _type
        for gnn in gnns:
            cnt += 1
            args.gnns = gnn
            key = "_".join([args.dataset, args.type, args.gnns])


            begin_time = time.time()
            time_local = time.localtime(int(begin_time))
            transform_time = time.strftime("%Y%m%d_%H%M%S", time_local)

            file_path = 'logs/' + args.dataset + '_' + args.type + '_' + args.gnns + '_' + str(
                args.seed) + '_' + transform_time + '.log'
            print(file_path, 'dataset:{}, type:{}, gnns:{}, times: {}/{}'.format(args.dataset, args.type, args.gnns, cnt, total))
            if key in finished:
                continue
            # f = open(file_path, 'a')
            # sys.stdout = f
            # sys.stderr = f

            sys.stdout = Logger(file_path)
            sys.stderr = Logger(file_path)

            print('cuda: %s' % args.cuda)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)

            if args.dataset == 'arxiv':
                pyg_data = PygNodePropPredDataset(name='ogbn-arxiv')
                data = Pyg2Dpr(pyg_data)
            else:
                data = Dataset(root='tmp/', name=args.dataset)
            adj, features, labels = data.adj, data.features, data.labels
            idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

            save_path = 'perturbations/' + args.type + '_' + args.dataset + '_results.txt'

            file = open(save_path, 'r')
            x = file.readlines()

            print('test read list')
            n = len(x)
            global_attack_perturbation_list = []
            for i in range(n):
                x[i] = x[i].strip()
                x[i] = x[i].strip("[]")
                x[i] = x[i].split(",")
                x[i] = list(map(int, x[i]))
                global_attack_perturbation_list.append(x[i])
                # print(x[i])

            print(global_attack_perturbation_list)

            if args.dataset == 'cora':
                target_node_list = [1152, 2429, 1762, 575, 2348, 1260, 820, 2143, 1591, 845, 69, 1298, 1198, 28, 1994,
                                    129, 290, 1686, 1217, 499, 1813, 1848, 1425, 1111, 160, 2266, 973, 1903, 1670, 2350,
                                    1761, 1267, 233, 773, 819, 2018, 298, 1137, 1337, 1019, 2263, 1288, 652, 866, 181,
                                    1005, 1147, 734, 1810, 265, 1274, 2421, 1741, 1599, 2064, 1833, 156, 732, 889, 2345,
                                    1028, 1592, 1120, 1909, 1359, 1691, 585, 2329, 1272, 506, 978, 1153, 880, 1673,
                                    2156, 757, 1491, 1526, 2226, 2254, 1433, 2335, 172, 1865, 2075, 2089, 2352, 877,
                                    2149, 235, 133, 1330, 2426, 1712, 964, 2455, 2466, 158, 1974, 1824, 1487, 2328, 88,
                                    1462, 270, 646, 2036, 2012, 1416, 1401, 613, 541, 368, 842, 569, 2087, 130, 435,
                                    2246, 1557, 2023, 1544, 1816, 1197, 503, 2407, 565, 612, 141, 1363, 850, 441, 1045,
                                    351, 2388, 2069, 1250, 669, 1823, 1116, 20, 452, 1625, 418, 1038, 2288, 2122, 296,
                                    513, 1950, 1435, 1396, 1229, 1703, 367, 1077, 594, 1612, 1635, 2439, 2476, 31, 1769,
                                    1255, 1997, 970, 832, 1292, 625, 477, 1432, 2324, 1380, 822, 955, 775, 1745, 1148,
                                    706, 1423, 2376, 838, 717, 883, 2397, 2427, 1665, 708, 2077, 2184, 2360, 1689, 1891,
                                    1308, 2465, 2112, 2188, 1486, 269, 64]
            elif args.dataset == 'pubmed':
                target_node_list = [16582, 19230, 6448, 5609, 8472, 16432, 9454, 18347, 14580, 18418, 5197, 4374, 16319,
                                    19687, 9730, 3606, 16100, 14075, 17295, 16640, 8379, 18408, 8059, 10934, 18802,
                                    14634, 10518, 18301, 8461, 9422, 16935, 15110, 13633, 6826, 9299, 6747, 1328, 8464,
                                    4449, 13344, 9103, 4605, 4985, 14206, 10337, 14964, 1733, 11968, 9516, 18883, 11887,
                                    789, 593, 10938, 17653, 3935, 19617, 13772, 13879, 17569, 1239, 19706, 18620, 18865,
                                    8140, 5689, 145, 8136, 458, 4258, 44, 13376, 19662, 10036, 16588, 12012, 16877,
                                    17906, 10733, 5592, 17329, 16741, 14803, 6672, 11457, 9275, 11487, 16454, 5296,
                                    1381, 13408, 8942, 15444, 18136, 2277, 12560, 764, 16865, 10475, 18042, 8261, 17660,
                                    2027, 17229, 7338, 6176, 12133, 8906, 11950, 2150, 4278, 659, 13068, 4034, 9852,
                                    13065, 12603, 14112, 2549, 17736, 1291, 2838, 1442, 1893, 4415, 6280, 17940, 10080,
                                    10826, 2884, 10012, 13070, 4532, 8743, 7803, 4123, 15559, 3270, 2222, 14043, 18037,
                                    9827, 17923, 10960, 13362, 5748, 901, 2049, 373, 11676, 1562, 6853, 898, 11687,
                                    8081, 5031, 17802, 1571, 1624, 5102, 6374, 18989, 11967, 17849, 18875, 306, 16446,
                                    7550, 7369, 9421, 8319, 10932, 4292, 18138, 7323, 3002, 17808, 1818, 5751, 14947,
                                    19182, 10819, 11083, 1833, 13909, 986, 4459, 13982, 12760, 1558, 10766, 893, 3895,
                                    15676, 2677, 14752, 2586, 3870, 10152, 9793]
            elif args.dataset == 'arxiv':
                target_node_list = [59052, 31662, 41835, 89882, 43334, 15799, 118734, 3660, 107108, 153315, 24945,
                                    44796, 75458, 26234, 88521, 82013, 99158, 85618, 164737, 12543, 18506, 27621, 46300,
                                    71500, 118871, 74159, 5932, 151605, 163893, 7549, 85981, 166107, 120997, 106674,
                                    30235, 159546, 61813, 114690, 156120, 18391, 15628, 140957, 114512, 81476, 34235,
                                    117245, 84604, 105603, 20788, 80182, 97833, 127691, 30173, 40630, 3368, 154150,
                                    24947, 78103, 1322, 129439, 115009, 145217, 111875, 17057, 58224, 143978, 157465,
                                    102938, 126831, 117654, 2536, 104337, 128064, 154657, 146840, 11091, 122872, 136223,
                                    52540, 25041, 112258, 157461, 130796, 142818, 73878, 34013, 150753, 56480, 120149,
                                    39643, 59132, 136933, 39375, 51634, 159489, 48524, 97268, 55189, 145389, 26418,
                                    104406, 104256, 81990, 166158, 34129, 150708, 8978, 77932, 36864, 167917, 138858,
                                    19336, 105566, 53998, 121510, 168848, 68355, 3317, 159521, 21055, 96564, 102573,
                                    30297, 107998, 28922, 56495, 169125, 109707, 137246, 107953, 94562, 162775, 21152,
                                    48497, 10098, 132207, 29067, 5003, 130758, 166624, 79070, 135639, 8387, 25004,
                                    43653, 12979, 161376, 45871, 22275, 142273, 49044, 145014, 125457, 125028, 76285,
                                    96423, 47539, 130935, 107934, 32847, 166562, 89434, 53817, 85909, 39550, 99013,
                                    28581, 48569, 86562, 85956, 40873, 149548, 128123, 146245, 72350, 76761, 71545,
                                    40981, 73991, 154202, 113165, 158728, 115950, 95133, 36881, 14396, 16668, 88543,
                                    165270, 137498, 133596, 13996, 141002, 71623, 137440, 112228, 10237, 71403, 57775,
                                    15615]
            elif args.dataset == 'citeseer':
                target_node_list = [1442, 1897, 1609, 514, 970, 1205, 93, 417, 1588, 398, 1958, 2045, 808, 703, 482,
                                    1233, 494, 1895, 702, 194, 793, 2032, 611, 1022, 1045, 181, 1714, 860, 694, 931, 55,
                                    682, 1315, 2062, 154, 915, 338, 2040, 1399, 1534, 657, 1974, 1014, 1576, 1994, 1732,
                                    1425, 716, 1975, 1160, 847, 131, 2049, 949, 1830, 377, 1817, 1899, 1212, 619, 1541,
                                    780, 266, 1970, 523, 1502, 481, 1508, 1686, 753, 1796, 355, 1860, 252, 978, 1108,
                                    1538, 147, 1310, 559, 1515, 1257, 438, 1594, 1790, 17, 1827, 311, 1793, 2016, 184,
                                    1699, 933, 791, 1691, 399, 106, 1353, 376, 938, 1398, 1270, 1670, 211, 446, 1198,
                                    2001, 419, 448, 1658, 404, 473, 838, 495, 785, 865, 920, 543, 1222, 964, 527, 1888,
                                    1976, 198, 280, 551, 1393, 663, 21, 737, 2063, 1736, 1820, 1551, 1930, 1735, 1687,
                                    1167, 834, 1298, 483, 296, 1159, 270, 2104, 299, 1878, 1085, 1401, 403, 348, 1838,
                                    950, 1700, 1640, 1092, 2029, 1390, 908, 1445, 452, 241, 1182, 1671, 1969, 1299, 803,
                                    968, 1696, 1991, 1840, 98, 541, 1738, 188, 1808, 1522, 275, 1443, 148, 1855, 1170,
                                    2079, 1007, 635, 1448, 777, 958, 1349, 956, 282, 1510, 168, 1828, 247, 1573, 2039,
                                    1260, 1973, 1003]

            # target_node_list = target_node_list[0:50]

            # # load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
            # print('==================')
            # print('=== load graph perturbed by Zugner metattack (under prognn splits) ===')
            # perturbed_data = PrePtbDataset(root='/tmp/',
            #         name=args.dataset,
            #         attack_method='meta',
            #         ptb_rate=args.ptb_rate)
            # perturbed_adj = perturbed_data.adj

            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)

            if args.gnns == 'gcn':
                if args.is_test == "true":
                    target_node_list = np.random.choice(target_node_list, args.test_nodes_nums, replace=False)


                # Setup GCN Model
                model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max() + 1, device=device)
                model = model.to(device)

                # Setup Surrogate model
                surrogate_gcn = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                                    nhid=16, device=device)
                surrogate_gcn = surrogate_gcn.to(device)
                surrogate_gcn.fit(features, adj, labels, idx_train, idx_val)

                original_output = surrogate_gcn.predict(features, adj)
                eva_cnt = 0
                poi_cnt = 0
                margin_decreases = []
                for i in range(len(target_node_list)):
                    print('{}/{}'.format(i, len(target_node_list)))
                    target_node = target_node_list[i]
                    current_adv_links = global_attack_perturbation_list[i]
                    perturbed_adj = copy.deepcopy(adj)
                    for j in range(len(current_adv_links)):
                        perturbed_adj[target_node, current_adv_links[j]] = 1 - perturbed_adj[
                            target_node, current_adv_links[j]]
                        perturbed_adj[current_adv_links[j], target_node] = 1 - perturbed_adj[
                            current_adv_links[j], target_node]

                    # evasion test
                    if args.is_eva == "true":
                        output = surrogate_gcn.predict(features, perturbed_adj)
                        acc_test = (output.argmax(1)[target_node] == labels[target_node])
                        eva_single_acc = acc_test.item()
                        eva_cnt += eva_single_acc

                    if args.is_test == "true":
                        original_output_target_node = original_output[target_node]
                        perturbed_output_target_node = output[target_node]
                        target_node_label = labels[target_node]
                        margin_decrease = original_output_target_node[target_node_label] - perturbed_output_target_node[target_node_label]
                        margin_decreases.append(margin_decrease.item())
                        print(margin_decreases)



                    # poi test
                    if args.is_poi == "true":
                        model.fit(features, perturbed_adj, labels, idx_train, train_iters=200, verbose=False)
                        # # using validation to pick model
                        # model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
                        model.eval()
                        # You can use the inner function of model to test

                        output = model.predict(features, perturbed_adj)
                        acc_test = (output.argmax(1)[target_node] == labels[target_node])

                        poi_single_acc = acc_test.item()
                        poi_cnt += poi_single_acc
            elif args.gnns == 'clustergcn':
                if args.is_test == "true":
                    target_node_list = np.random.choice(target_node_list, args.test_nodes_nums, replace=False)

                import graphgallery as gg
                from graphgallery.datasets import NPZDataset
                import graphgallery.functional as gf
                from graphgallery.gallery.nodeclas import ClusterGCN

                gg.set_backend("pytorch")
                data = NPZDataset(args.dataset,
                                  root="/tmp/",
                                  verbose=False,
                                  transform="standardize")
                graph = data.graph
                print(data.root)
                # gf.random_seed(args.seed, gg.backend())
                # splits = data.split_nodes(random_state=15)
                # print(splits.train_nodes)
                # print(idx_train)

                device = "gpu" if torch.cuda.is_available() else "cpu"
                # Setup GCN Model
                # model = ClusterGCN(device="cpu", seed=args.seed).setup_graph(graph, num_clusters=10, attr_transform="normalize_attr").build(hids=[16], acts=['relu'], dropout=0.5, lr=0.01)

                # Setup Surrogate model
                surrogate_gcn = ClusterGCN(device=device, seed=args.seed).setup_graph(graph, num_clusters=10, attr_transform="normalize_attr").build(hids=[16], acts=['relu'], dropout=0.5, lr=0.01)
                surrogate_gcn.fit(idx_train, idx_val, verbose=1, epochs=200)
                # _results = surrogate_gcn.evaluate(idx_test)
                # print(f'Test loss {_results.loss:.5}, Test accuracy {_results.accuracy:.2%}')

                original_output = surrogate_gcn.predict(np.arange(adj.shape[0]))
                original_output = gf.get('softmax')(original_output)

                eva_cnt = 0
                poi_cnt = 0
                margin_decreases = []
                for i in range(len(target_node_list)):
                    print('{}/{}'.format(i, len(target_node_list)))
                    target_node = target_node_list[i]
                    current_adv_links = global_attack_perturbation_list[i]
                    perturbed_adj = copy.deepcopy(adj)
                    for j in range(len(current_adv_links)):
                        perturbed_adj[target_node, current_adv_links[j]] = 1 - perturbed_adj[
                            target_node, current_adv_links[j]]
                        perturbed_adj[current_adv_links[j], target_node] = 1 - perturbed_adj[
                            current_adv_links[j], target_node]
                    tmp_graph = graph.copy()
                    tmp_graph.adj_matrix = perturbed_adj
                    # evasion test
                    if args.is_eva == "true":
                        surrogate_gcn.setup_graph(tmp_graph, num_clusters=10, attr_transform="normalize_attr")
                        output = surrogate_gcn.predict(np.arange(perturbed_adj.shape[0]))
                        output = gf.get('softmax')(original_output)
                        acc_test = (output.argmax(1)[target_node] == labels[target_node])
                        eva_single_acc = acc_test.item()
                        eva_cnt += eva_single_acc

                    if args.is_test == "true":
                        original_output_target_node = original_output[target_node]
                        perturbed_output_target_node = output[target_node]
                        target_node_label = labels[target_node]
                        margin_decrease = original_output_target_node[target_node_label] - perturbed_output_target_node[
                            target_node_label]
                        margin_decreases.append(margin_decrease)
                        # print(margin_decreases)

                    # poi test
                    if args.is_poi == "true":
                        poi_model = ClusterGCN(device=device, seed=args.seed).setup_graph(graph, num_clusters=10, attr_transform="normalize_attr").build(hids=[16], acts=['relu'], dropout=0.5, lr=0.01)
                        poi_model.setup_graph(tmp_graph, num_clusters=10, attr_transform="normalize_attr")
                        poi_model.fit(idx_train, idx_val, verbose=1, epochs=200)
                        output = poi_model.predict(np.arange(perturbed_adj.shape[0]))
                        output = gf.get('softmax')(original_output)
                        acc_test = (output.argmax(1)[target_node] == labels[target_node])

                        poi_single_acc = acc_test.item()
                        poi_cnt += poi_single_acc

            eva_asr = 0
            poi_asr = 0
            avg_decrease = 0
            if args.is_eva == "true":
                eva_asr = 1 - eva_cnt / len(target_node_list)
            if args.is_poi == "true":
                poi_asr = 1 - poi_cnt / len(target_node_list)
            if args.is_test == "true":
                avg_decrease = np.mean(margin_decreases)
            cur_result = [eva_asr, poi_asr, avg_decrease]
            results.loc[key] = cur_result
            results.to_csv('kdd_test_' + total_time + '.csv')
            print(results)
