import graph


def get_list_nodes_and_edges_from_file(name_file):
    list_all = []
    li_nodes = []
    li_edges = []
    with open(name_file, "r", encoding="utf-8") as f:
        for line in f:
            list_all.append(line)
        f.close()
        for i in range(len(list_all)):
            if 'id' in list_all[i]:
                li_nodes.append(int(list_all[i][7:]))
            if 'source' in list_all[i]:
                source = int(list_all[i][11:])
                i += 1
                target = int(list_all[i][11:])
                i += 1
                weight = float(list_all[i][11:])
                edge = (source, target, weight)
                li_edges.append(edge)

    return li_nodes, li_edges


def get_G_from_lists(li_nodes, li_edges):
    G = graph.Graph()
    G.add_nodes_from(li_nodes)
    G.add_weighted_edges_from(li_edges)
    return G


def get_G_from_file(name_file):
    li_nodes, li_edges = get_list_nodes_and_edges_from_file(name_file)
    return get_G_from_lists(li_nodes, li_edges)
