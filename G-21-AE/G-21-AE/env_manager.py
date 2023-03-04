import networkx as nx
import graph
import file_process
from env import Env


class EnvManager:
    def __init__(self, list_file_graph, use_gpu=False):
        self.list_graph = []
        self.list_env = []
        self.use_gpu = use_gpu
        for i in list_file_graph:
            G = file_process.get_G_from_file(name_file=i)
            if self.use_gpu:
                E = Env(G, use_gpu=self.use_gpu)
            else:
                E = Env(G)
            self.list_graph.append(G)
            self.list_env.append(E)
