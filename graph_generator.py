# File: graph_generator.py
import random
import networkx as nx

def generate_tree(config: dict):
    max_depth = config['max_depth']
    min_children = config['min_children']
    max_children = config['max_children']

    G = nx.DiGraph()
    node_id = 0
    stack = [(node_id, 0)] 
    G.add_node(node_id, depth=0)
    depth_dict = {0: [node_id]}
    node_id += 1

 
    while stack:
        parent, depth = stack.pop()
        if depth >= max_depth:
            continue

        num_children = random.randint(min_children, max_children)
        for _ in range(num_children):
            child = node_id
            G.add_node(child, depth=depth + 1)
            G.add_edge(parent, child)
            stack.append((child, depth + 1))
            depth_dict.setdefault(depth + 1, []).append(child)
            node_id += 1

   
    for d1 in depth_dict:
        for d2 in range(d1 + 2, max_depth + 1): 
            for from_node in depth_dict[d1]:
                for to_node in random.sample(depth_dict[d2], k=min(1, len(depth_dict[d2]))): 
                    if not G.has_edge(from_node, to_node) and nx.has_path(G.reverse(), to_node, from_node) is False:
                        G.add_edge(from_node, to_node)

    for node in G.nodes:
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        if in_deg == 0:
            G.nodes[node]['type'] = 'root'
        elif out_deg == 0:
            G.nodes[node]['type'] = 'leaf'
        else:
            G.nodes[node]['type'] = 'intermediate'

    return G