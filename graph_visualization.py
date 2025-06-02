import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G, filename=None):
    pos = nx.spring_layout(G)
    node_colors = []

    for node in G.nodes(data=True):
        if node[1]['type'] == 'root':
            node_colors.append('lightgreen')
        elif node[1]['type'] == 'intermediate':
            node_colors.append('lightblue')
        else:
            node_colors.append('salmon')

    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_size=10)
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
