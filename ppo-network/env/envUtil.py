import torch
from node2vec import Node2Vec
from sklearn.decomposition import PCA


def graph_embedding_to_1dim(G, use_gpu=False):
    # 设置node2vec参数
    node2vec = Node2Vec(G,
                        dimensions=32,  # 嵌入维度
                        p=1,  # 回家参数
                        q=3,  # 外出参数
                        walk_length=10,  # 随机游走最大长度
                        num_walks=600,  # 每个节点作为起始节点生成的随机游走个数
                        workers=32,  # 并行线程数
                        quiet=True
                        )

    # p=1, q=0.5, n_clusters=6。DFS深度优先搜索，挖掘同质社群
    # p=1, q=2, n_clusters=3。BFS宽度优先搜索，挖掘节点的结构功能。

    model = node2vec.fit(window=3,  # Skip-Gram窗口大小
                         min_count=1,  # 忽略出现次数低于此阈值的节点（词）
                         batch_words=48  # 每个线程处理的数据量
                         )
    G_ndarray = model.wv.vectors
    pca = PCA(n_components=1)
    if use_gpu:
        return torch.FloatTensor(pca.fit_transform(G_ndarray)).cuda().T
    tensor = torch.FloatTensor(pca.fit_transform(G_ndarray))
    return tensor.T


def calculate_ANC(list_connectivity, origin_connectivity, num_origin_nodes):
    return sum(list_connectivity) / (origin_connectivity * num_origin_nodes)
