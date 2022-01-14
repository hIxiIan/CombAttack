import networkx as nx


class tGraph(object):
    def __init__(self, file_='dataset/phishing/TransEdgelist.txt', verbose=0):
        self.G = nx.MultiDiGraph()

        if verbose > 0:
            print("Loading file", file_, "...")
        edge_key = 0

        with open(file_) as f:
            for l in f:
                x, y, a, t = l.strip().split(',')
                a = float(a)
                t = int(t)
                x = str(int(x) - 1)
                y = str(int(y) - 1)
                if self.G.has_edge(x, y, t):
                    if self.G[x][y][t]['weight'] != a:
                        self.G[x][y][t]['weight'] += a
                else:
                    self.G.add_edge(x, y, key=t, weight=a)
                edge_key = edge_key + 1

        self.number_of_nodes = self.G.number_of_nodes()
        self.number_of_edges = self.G.number_of_edges()
        if verbose > 0:
            print("Summary of graph:")
            print("Number of nodes: ", self.number_of_nodes)
            print("Number of edges: ", self.number_of_edges)
            print("Number of edge_key: ", edge_key)