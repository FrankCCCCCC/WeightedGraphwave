
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
def cycle(start, len_cycle, role_start=0, plot=False):
    '''Builds a cycle graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    plot        :    boolean -- should the shape be printed?
    OUTPUT:
    -------------
    graph           :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    '''
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 10))
    graph.add_weighted_edges_from([(start    , start + 1, 1.0)])
    graph.add_weighted_edges_from([(start    , start + 2, 5.0)])
    graph.add_weighted_edges_from([(start    , start + 3, 4.0)])
    graph.add_weighted_edges_from([(start    , start + 4, 4.0)])
    graph.add_weighted_edges_from([(start    , start + 5, 3.0)])
    graph.add_weighted_edges_from([(start    , start + 6, 3.0)])
    graph.add_weighted_edges_from([(start    , start + 7, 2.0)])
    graph.add_weighted_edges_from([(start    , start + 8, 2.0)])
    graph.add_weighted_edges_from([(start    , start + 9, 1.0)])
    graph.add_weighted_edges_from([(start + 1, start + 2, 1.0)])
    graph.add_weighted_edges_from([(start + 1, start + 3, 1.0)])
    graph.add_weighted_edges_from([(start + 1, start + 4, 1.0)])
    graph.add_weighted_edges_from([(start + 1, start + 5, 1.0)])
    graph.add_weighted_edges_from([(start + 1, start + 6, 1.0)])
    graph.add_weighted_edges_from([(start + 1, start + 7, 1.0)])
    graph.add_weighted_edges_from([(start + 1, start + 8, 1.0)])
    graph.add_weighted_edges_from([(start + 1, start + 9, 1.0)])
    graph.add_weighted_edges_from([(start + 2, start + 3, 1.0)])
    graph.add_weighted_edges_from([(start + 2, start + 4, 1.0)])
    graph.add_weighted_edges_from([(start + 2, start + 5, 1.0)])
    graph.add_weighted_edges_from([(start + 2, start + 6, 1.0)])
    graph.add_weighted_edges_from([(start + 2, start + 7, 1.0)])
    graph.add_weighted_edges_from([(start + 2, start + 8, 1.0)])
    graph.add_weighted_edges_from([(start + 2, start + 9, 1.0)])
    graph.add_weighted_edges_from([(start + 3, start + 4, 1.0)])
    graph.add_weighted_edges_from([(start + 3, start + 5, 1.0)])
    graph.add_weighted_edges_from([(start + 3, start + 6, 1.0)])
    graph.add_weighted_edges_from([(start + 3, start + 7, 1.0)])
    graph.add_weighted_edges_from([(start + 3, start + 8, 1.0)])
    graph.add_weighted_edges_from([(start + 3, start + 9, 1.0)])
    graph.add_weighted_edges_from([(start + 4, start + 5, 1.0)])
    graph.add_weighted_edges_from([(start + 4, start + 6, 1.0)])
    graph.add_weighted_edges_from([(start + 4, start + 7, 1.0)])
    graph.add_weighted_edges_from([(start + 4, start + 8, 1.0)])
    graph.add_weighted_edges_from([(start + 4, start + 9, 1.0)])
    graph.add_weighted_edges_from([(start + 5, start + 6, 1.0)])
    graph.add_weighted_edges_from([(start + 5, start + 7, 1.0)])
    graph.add_weighted_edges_from([(start + 5, start + 8, 1.0)])
    graph.add_weighted_edges_from([(start + 5, start + 9, 1.0)])
    graph.add_weighted_edges_from([(start + 6, start + 7, 1.0)])
    graph.add_weighted_edges_from([(start + 6, start + 8, 1.0)])
    graph.add_weighted_edges_from([(start + 6, start + 9, 1.0)])
    graph.add_weighted_edges_from([(start + 7, start + 8, 1.0)])
    graph.add_weighted_edges_from([(start + 7, start + 9, 1.0)])
    graph.add_weighted_edges_from([(start + 8, start + 9, 1.0)])
    roles = [role_start] * 10
    if plot is True: plot_networkx(graph, roles)
    return graph, roles
def plot_networkx(graph, role_labels):
        cmap = plt.get_cmap('hot')
        x_range = np.linspace(0, 1, len(np.unique(role_labels)))
        coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_labels))}
        node_color = [coloring[role_labels[i]] for i in range(len(role_labels))]
        plt.figure()
        nx.draw_networkx(graph, pos=nx.layout.fruchterman_reingold_layout(graph),
                         node_color=node_color, cmap='hot')
        plt.show()
        return
#cycle(0,16,role_start=0,plot=True)