from basic_sampler import BasicSampler
import random


class RandomSampler(BasicSampler):
    def __init__(self, graph, seed):
        super().__init__(graph, seed=seed)

    def sample(self, sample_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        graph = self.graph
        nodes = [start_node]

        while len(nodes) < sample_length:
            cur = nodes[-1]
            cur_nbrs = list(graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                nodes.append(random.choice(cur_nbrs))
            else:
                break
        return nodes

    def simulate_samples(self, sample_length, start_nodes):
        walks = []
        print('start_nodes iteration:')
        for start_node in start_nodes:
            print('start node:', start_node)
            walks.append(self.deepwalk_walk(sample_length=sample_length, start_node=start_node))
        return walks
