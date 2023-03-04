import networkx as nx
from envUtil import graph_embedding_to_1dim
from envUtil import calculate_ANC
import env_manager


class Env:
    def __init__(self, G,use_gpu=False):
        self.G = G
        self.origin_G = self.G.copy()
        self.num_nodes = self.G.number_of_nodes()
        self.actions = self.G.number_of_nodes()
        self.use_gpu=use_gpu
        self.obs = graph_embedding_to_1dim(G, use_gpu=self.use_gpu)
        self.origin_connectivity = nx.average_node_connectivity(G=self.G)
        # self.list_connectivity = []

    def step(self, action):
        done = 0
        # ANC_origin = calculate_ANC(list_connectivity=self.list_connectivity,
        #                            origin_connectivity=self.origin_connectivity,
        #                            num_origin_nodes=self.num_origin_nodes)
        connectivity_before = nx.average_node_connectivity(G=self.G)
        self.G.remove_node(action)
        self.G.add_node(action)
        self.obs = graph_embedding_to_1dim(self.G, use_gpu=self.use_gpu)
        next_obs = self.obs
        average_node_connectivity = nx.average_node_connectivity(self.G)
        # self.list_connectivity.append(average_node_connectivity)
        # ANC_after = calculate_ANC(list_connectivity=self.list_connectivity,
        #                           origin_connectivity=self.origin_connectivity,
        #                           num_origin_nodes=self.num_origin_nodes)
        connectivity_after = nx.average_node_connectivity(G=self.G)
        reward = 100 * (connectivity_before - connectivity_after)
        if average_node_connectivity == 0.0:
            done = 1
        return next_obs, reward, done

    def reset(self):
        self.G = self.origin_G.copy()
        self.obs = graph_embedding_to_1dim(self.G, use_gpu=self.use_gpu)
        return self.obs



if __name__ == '__main__':
    list_file_graph = ['g_27']
    em = env_manager.EnvManager(list_file_graph=list_file_graph)
    env = em.list_env[0]
    for i in range(env.G.number_of_nodes()):
        next_obs, reward, done = env.step(i)
        print(next_obs)
        print(reward)
        print(done)
        print('------------------------------------------------------')
