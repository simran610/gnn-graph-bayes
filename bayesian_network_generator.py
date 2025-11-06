# import random
# import networkx as nx
# import numpy as np
# from pgmpy.models import BayesianNetwork
# from pgmpy.factors.discrete import TabularCPD

# def generate_varied_bn(config):
#     random.seed(config['random_seed'])
#     min_depth = config.get('min_depth', 3)
#     max_depth = config.get('max_depth', 10)
#     min_children = config.get('min_children', 1)
#     max_children = config.get('max_children', 4)
#     max_nodes = config.get('max_nodes', 200)

#     # Choose actual depth randomly within range
#     actual_depth = random.randint(min_depth, max_depth)

#     G = nx.DiGraph()
#     node_id = 0
#     G.add_node(node_id, depth=0)
#     frontier = [(node_id, 0)]  # Node and its depth
#     node_id += 1

#     # Hybrid BFS-DFS: Use a queue (BFS) but also randomize children count
#     while frontier and len(G) < max_nodes:
#         parent, depth = frontier.pop(0)
#         if depth >= actual_depth:
#             continue

#         num_children = random.randint(min_children, max_children)
#         for _ in range(num_children):
#             if len(G) >= max_nodes:
#                 break
#             child = node_id
#             G.add_node(child, depth=depth + 1)
#             G.add_edge(parent, child)
#             frontier.append((child, depth + 1))
#             node_id += 1

#     # Assign node types for compatibility (root, leaf, intermediate)
#     for node in G.nodes:
#         in_deg = G.in_degree(node)
#         out_deg = G.out_degree(node)
#         if in_deg == 0:
#             G.nodes[node]['type'] = 'root'
#         elif out_deg == 0:
#             G.nodes[node]['type'] = 'leaf'
#         else:
#             G.nodes[node]['type'] = 'intermediate'

#     # Build BN with CPDs similar to previous code
#     model = BayesianNetwork()
#     model.add_edges_from(G.edges())
#     card = 2  # binary
#     for node in G.nodes():
#         parents = list(G.predecessors(node))
#         if not parents:
#             root_prob = np.random.beta(2, 5) if np.random.random() < 0.5 else np.random.beta(5, 2)
#             cpd = TabularCPD(variable=node, variable_card=card, values=[[1-root_prob], [root_prob]])
#         else:
#             parent_card = [card] * len(parents)
#             num_cols = card ** len(parents)
#             values = np.zeros((card, num_cols))
#             for col in range(num_cols):
#                 parent_config = [(col >> i) & 1 for i in range(len(parents))]
#                 parent_sum = sum(parent_config)
#                 base_prob = 0.3 + 0.4 * parent_sum / len(parents)
#                 noise = np.random.normal(0, 0.2)
#                 prob_true = np.clip(base_prob + noise, 0.1, 0.9)
#                 values[0, col] = 1 - prob_true
#                 values[1, col] = prob_true
#             cpd = TabularCPD(variable=node, variable_card=card, values=values.tolist(), evidence=parents, evidence_card=parent_card)
#         model.add_cpds(cpd)

#     assert model.check_model()
#     return G, model

# import networkx as nx
# import numpy as np
# import random
# from pgmpy.models import BayesianNetwork
# from pgmpy.factors.discrete import TabularCPD

# def generate_varied_bn(config):
#     random.seed(config['random_seed'])
#     np.random.seed(config['random_seed'])
    
#     min_depth = config.get('min_depth', 3)
#     max_depth = config.get('max_depth', 10)
#     min_children = config.get('min_children', 1)
#     max_children = config.get('max_children', 4)
#     max_nodes = config.get('max_nodes', 200)
#     min_nodes = config.get('min_nodes', 6)  # Add this to your config
    
#     # Sample target number of nodes from a distribution
#     # This creates variety - some small graphs, some large, balanced distribution
#     target_nodes = random.randint(min_nodes, max_nodes)
    
#     # Choose actual depth randomly within range
#     actual_depth = random.randint(min_depth, max_depth)

#     G = nx.DiGraph()
#     node_id = 0
#     G.add_node(node_id, depth=0)
#     frontier = [(node_id, 0)]  # Node and its depth
#     node_id += 1

#     # Stop at target_nodes, not max_nodes
#     while frontier and len(G) < target_nodes:
#         parent, depth = frontier.pop(0)
        
#         # Stop expanding if we've reached target depth
#         if depth >= actual_depth:
#             continue
        
#         # Dynamically adjust children based on remaining nodes
#         remaining_nodes = target_nodes - len(G)
#         if remaining_nodes <= 0:
#             break
            
#         # Limit children count to not overshoot target
#         max_possible_children = min(max_children, remaining_nodes)
#         if max_possible_children < min_children:
#             num_children = max_possible_children
#         else:
#             num_children = random.randint(min_children, max_possible_children)
        
#         for _ in range(num_children):
#             if len(G) >= target_nodes:
#                 break
#             child = node_id
#             G.add_node(child, depth=depth + 1)
#             G.add_edge(parent, child)
#             frontier.append((child, depth + 1))
#             node_id += 1

#     # Assign node types for compatibility (root, leaf, intermediate)
#     for node in G.nodes:
#         in_deg = G.in_degree(node)
#         out_deg = G.out_degree(node)
#         if in_deg == 0:
#             G.nodes[node]['type'] = 'root'
#         elif out_deg == 0:
#             G.nodes[node]['type'] = 'leaf'
#         else:
#             G.nodes[node]['type'] = 'intermediate'

#     # Build BN with CPDs
#     model = BayesianNetwork()
#     model.add_edges_from(G.edges())
    
#     # Variable cardinality for more varied PCPD
#     # Instead of always 2, randomly choose cardinality per node
#     node_cardinalities = {}
#     for node in G.nodes():
#         # Vary between 2-5 states per variable to get different PCPD values
#         node_cardinalities[node] = random.choice([2, 3, 4, 5])
    
#     for node in G.nodes():
#         card = node_cardinalities[node]
#         parents = list(G.predecessors(node))
        
#         if not parents:
#             # Root node - uniform distribution across states
#             probs = np.random.dirichlet(np.ones(card))
#             cpd = TabularCPD(
#                 variable=node, 
#                 variable_card=card, 
#                 values=probs.reshape(-1, 1).tolist()
#             )
#         else:
#             parent_card = [node_cardinalities[p] for p in parents]
#             num_cols = np.prod(parent_card)
#             values = np.zeros((card, num_cols))
            
#             for col in range(num_cols):
#                 # Generate random probability distribution for each parent configuration
#                 probs = np.random.dirichlet(np.ones(card))
#                 values[:, col] = probs
            
#             cpd = TabularCPD(
#                 variable=node, 
#                 variable_card=card, 
#                 values=values.tolist(), 
#                 evidence=parents, 
#                 evidence_card=parent_card
#             )
        
#         model.add_cpds(cpd)

#     assert model.check_model()
#     return G, model

import networkx as nx
import numpy as np
import random
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def generate_varied_bn_fixed_cycle_check(config):
    """
    FIXED: Generates a DAG structured Bayesian Network with guaranteed cycle-free 
    edges and includes the necessary 'type' attribute for the exporter.
    """
    # --- 0. Configuration and Initialization ---
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Structure parameters
    min_nodes = config.get('min_nodes', 6)
    max_nodes = config.get('max_nodes', 200)
    min_children = config.get('min_children', 1)
    max_children = config.get('max_children', 4)
    min_depth = config.get('min_depth', 3)
    max_depth = config.get('max_depth', 10)
    
    # Complexity and Cardinality parameters
    max_extra_edges_ratio = config.get('max_extra_edges_ratio', 0.15) 
    max_cardinality = config.get('max_cardinality', 7) 
    
    target_nodes = random.randint(min_nodes, max_nodes)
    actual_depth = random.randint(min_depth, max_depth)

    G = nx.DiGraph()
    node_id = 0
    G.add_node(node_id)
    frontier = [node_id]
    node_id += 1
    
    # --- 1. Structure Generation (Tree-like Core) ---
    while frontier and len(G) < target_nodes:
        parent = frontier.pop(0)
        
        if parent >= actual_depth * max_children: 
             continue

        remaining_nodes = target_nodes - len(G)
        max_possible_children = min(max_children, remaining_nodes)
        
        if max_possible_children >= min_children:
            num_children = random.randint(min_children, max_possible_children)
            for _ in range(num_children):
                if len(G) >= target_nodes:
                    break
                child = node_id
                G.add_node(child)
                G.add_edge(parent, child)
                frontier.append(child)
                node_id += 1

    # --- 2. Complexity Injection (Robust DAG Generation) ---
    max_extra_edges = int(len(G.edges()) * max_extra_edges_ratio)
    
    for _ in range(max_extra_edges):
        potential_child = random.choice(list(G.nodes()))
        candidates = list(G.nodes - {potential_child})
        
        if not candidates:
            continue

        random.shuffle(candidates)

        for new_parent in candidates:
            if G.has_edge(new_parent, potential_child):
                continue
            
            # Check for cycle using the robust nx.has_path
            if not nx.has_path(G, potential_child, new_parent):
                 G.add_edge(new_parent, potential_child)
                 break 

    # --- NEW: 2b. Assign Node Types for Exporter (FIX for KeyError) ---
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        
        if in_deg == 0:
            G.nodes[node]['type'] = 'root'
        elif out_deg == 0:
            G.nodes[node]['type'] = 'leaf'
        else:
            G.nodes[node]['type'] = 'intermediate'

    # --- 3. Cardinality Assignment ---
    model = BayesianNetwork()
    model.add_edges_from(G.edges())
    
    node_cardinalities = {}
    for node in G.nodes():
        node_cardinalities[node] = random.randint(2, max_cardinality)
    
    # --- 4. CPD Population (Dirichlet Distribution) ---
    for node in G.nodes():
        card = node_cardinalities[node]
        parents = list(G.predecessors(node))
        
        if not parents:
            # Root Node: Varied Dirichlet concentration for sparse/uniform priors.
            concentration = random.choice([0.1, 0.5, 1.0, 5.0]) 
            probs = np.random.dirichlet(np.ones(card) * concentration)
            probs = np.clip(probs, 1e-6, 1.0)
            probs /= probs.sum() 
            
            cpd = TabularCPD(variable=node, variable_card=card, values=probs.reshape(-1, 1).tolist())
            
        else:
            # Non-Root Node: General CPD generation
            parent_card = [node_cardinalities[p] for p in parents]
            num_cols = np.prod(parent_card)
            values = np.zeros((card, num_cols))
            
            for col in range(num_cols):
                concentration = random.choice([0.1, 1.0, 5.0])
                probs = np.random.dirichlet(np.ones(card) * concentration)

                probs = np.clip(probs, 1e-6, 1.0)
                probs /= probs.sum()
                
                values[:, col] = probs
            
            cpd = TabularCPD(variable=node, variable_card=card, values=values.tolist(), 
                             evidence=parents, evidence_card=parent_card)
        
        model.add_cpds(cpd)

    assert model.check_model()
    return G, model