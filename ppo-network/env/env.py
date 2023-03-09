from env.envUtil import graph_embedding_to_1dim
import numpy as np
import networkx as nx


class Env:
    def __init__(self, G, use_gpu=True):
        self.G = G
        self.origin_G = self.G.copy()
        self.num_nodes = self.G.number_of_nodes()
        self.actions = self.G.number_of_nodes()
        self.use_gpu = use_gpu
        self.observation_space = graph_embedding_to_1dim(G, use_gpu=self.use_gpu).cpu().numpy()
        self.origin_connectivity = nx.average_node_connectivity(G=self.G)
        self.action_space = self.observation_space
        self._max_episode_steps = self.num_nodes
        # self.list_connectivity = []

    def step(self, action):
        done = 0
        # ANC_origin = calculate_ANC(list_connectivity=self.list_connectivity,
        #                            origin_connectivity=self.origin_connectivity,
        #                            num_origin_nodes=self.num_origin_nodes)
        connectivity_before = nx.average_node_connectivity(G=self.G)
        action = np.argmax(action)
        self.G.remove_node(action)
        self.G.add_node(action)
        self.observation_space = graph_embedding_to_1dim(self.G, use_gpu=self.use_gpu)
        next_obs = self.observation_space
        # self.list_connectivity.append(average_node_connectivity)
        # ANC_after = calculate_ANC(list_connectivity=self.list_connectivity,
        #                           origin_connectivity=self.origin_connectivity,
        #                           num_origin_nodes=self.num_origin_nodes)
        connectivity_after = nx.average_node_connectivity(G=self.G)
        reward = 100 * (connectivity_before - connectivity_after)
        if connectivity_after == 0.0:
            done = 1
        return next_obs, reward, done, None

    def reset(self):
        self.G = self.origin_G.copy()
        self.observation_space = graph_embedding_to_1dim(self.G, use_gpu=self.use_gpu)
        return self.observation_space
