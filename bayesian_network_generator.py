import random
import networkx as nx
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def generate_varied_bn(config):
    random.seed(config['random_seed'])
    min_depth = config.get('min_depth', 3)
    max_depth = config.get('max_depth', 10)
    min_children = config.get('min_children', 1)
    max_children = config.get('max_children', 4)
    max_nodes = config.get('max_nodes', 200)

    # Choose actual depth randomly within range
    actual_depth = random.randint(min_depth, max_depth)

    G = nx.DiGraph()
    node_id = 0
    G.add_node(node_id, depth=0)
    frontier = [(node_id, 0)]  # Node and its depth
    node_id += 1

    # Hybrid BFS-DFS: Use a queue (BFS) but also randomize children count
    while frontier and len(G) < max_nodes:
        parent, depth = frontier.pop(0)
        if depth >= actual_depth:
            continue

        num_children = random.randint(min_children, max_children)
        for _ in range(num_children):
            if len(G) >= max_nodes:
                break
            child = node_id
            G.add_node(child, depth=depth + 1)
            G.add_edge(parent, child)
            frontier.append((child, depth + 1))
            node_id += 1

    # Assign node types for compatibility (root, leaf, intermediate)
    for node in G.nodes:
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        if in_deg == 0:
            G.nodes[node]['type'] = 'root'
        elif out_deg == 0:
            G.nodes[node]['type'] = 'leaf'
        else:
            G.nodes[node]['type'] = 'intermediate'

    # Build BN with CPDs similar to previous code
    model = BayesianNetwork()
    model.add_edges_from(G.edges())
    card = 2  # binary
    for node in G.nodes():
        parents = list(G.predecessors(node))
        if not parents:
            root_prob = np.random.beta(2, 5) if np.random.random() < 0.5 else np.random.beta(5, 2)
            cpd = TabularCPD(variable=node, variable_card=card, values=[[1-root_prob], [root_prob]])
        else:
            parent_card = [card] * len(parents)
            num_cols = card ** len(parents)
            values = np.zeros((card, num_cols))
            for col in range(num_cols):
                parent_config = [(col >> i) & 1 for i in range(len(parents))]
                parent_sum = sum(parent_config)
                base_prob = 0.3 + 0.4 * parent_sum / len(parents)
                noise = np.random.normal(0, 0.2)
                prob_true = np.clip(base_prob + noise, 0.1, 0.9)
                values[0, col] = 1 - prob_true
                values[1, col] = prob_true
            cpd = TabularCPD(variable=node, variable_card=card, values=values.tolist(), evidence=parents, evidence_card=parent_card)
        model.add_cpds(cpd)

    assert model.check_model()
    return G, model
