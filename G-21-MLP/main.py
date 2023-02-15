import file_process
import networkx as nx
import matplotlib.pyplot as plt


def draw(Graph):
    pos = nx.spring_layout(Graph, iterations=20)  # 我们设算法迭代次数为20次
    nx.draw_networkx_edges(Graph, pos, width=[float(d['weight'] * 2) for (u, v, d) in Graph.edges(data=True)],
                           edge_color="black")
    nx.draw_networkx_nodes(Graph, pos, node_color="black")
    # nx.draw_networkx_labels(Graph, pos, font_size=13, font_color='white')
    plt.show()


if __name__ == '__main__':
    G = file_process.get_G_from_file('g_27')

    print(nx.average_node_connectivity(G))
