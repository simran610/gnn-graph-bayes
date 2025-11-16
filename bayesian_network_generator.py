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

def generate_varied_bn(config):
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    min_depth = config.get('min_depth', 3)
    max_depth = config.get('max_depth', 10)
    min_children = config.get('min_children', 1)
    max_children = config.get('max_children', 4)
    max_nodes = config.get('max_nodes', 200)
    min_nodes = config.get('min_nodes', 6)
    
    # Sample target number of nodes
    target_nodes = random.randint(min_nodes, max_nodes)
    actual_depth = random.randint(min_depth, max_depth)

    # Generate graph structure
    G = nx.DiGraph()
    node_id = 0
    G.add_node(node_id, depth=0)
    frontier = [(node_id, 0)]
    node_id += 1

    while frontier and len(G) < target_nodes:
        parent, depth = frontier.pop(0)
        if depth >= actual_depth:
            continue
        
        remaining_nodes = target_nodes - len(G)
        if remaining_nodes <= 0:
            break
            
        max_possible_children = min(max_children, remaining_nodes)
        if max_possible_children < min_children:
            num_children = max_possible_children
        else:
            num_children = random.randint(min_children, max_possible_children)
        
        for _ in range(num_children):
            if len(G) >= target_nodes:
                break
            child = node_id
            G.add_node(child, depth=depth + 1)
            G.add_edge(parent, child)
            frontier.append((child, depth + 1))
            node_id += 1

    # Assign node types
    for node in G.nodes:
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        if in_deg == 0:
            G.nodes[node]['type'] = 'root'
        elif out_deg == 0:
            G.nodes[node]['type'] = 'leaf'
        else:
            G.nodes[node]['type'] = 'intermediate'

    # Build BN with realistic CPDs
    model = BayesianNetwork()
    model.add_edges_from(G.edges())
    
    # Assign cardinalities - mostly binary like real BNs (Asia, Child, Alarm)
    node_cardinalities = {}
    for node in G.nodes():
        # 70% binary, 20% ternary, 10% more states (like real BNs)
        rand = random.random()
        if rand < 0.7:
            node_cardinalities[node] = 2
        elif rand < 0.9:
            node_cardinalities[node] = 3
        else:
            node_cardinalities[node] = random.choice([4, 5])
    
    for node in G.nodes():
        card = node_cardinalities[node]
        parents = list(G.predecessors(node))
        
        if not parents:
            # Root nodes: Create diverse base rates (not uniform!)
            prob_type = random.random()
            if prob_type < 0.6:  # Rare events (like diseases)
                # probs = np.random.beta(0.5, 5, size=card)
                probs = np.random.beta(0.1, 10, size=card)
            elif prob_type < 0.8:  # Common events
                probs = np.random.beta(5, 2, size=card)
                
            else:  # Moderate probability
                probs = np.random.beta(2, 2, size=card)
            
            probs = probs / probs.sum()  # Normalize
            cpd = TabularCPD(
                variable=node, 
                variable_card=card, 
                values=probs.reshape(-1, 1).tolist()
            )
        else:
            parent_card = [node_cardinalities[p] for p in parents]
            num_cols = int(np.prod(parent_card))
            values = np.zeros((card, num_cols))
            
            # Choose a CPD pattern (like real Bayesian networks!)
            cpd_pattern = random.choice([
                'noisy_or',      # 25% - Common in medical BNs
                'deterministic', # 15% - Strong causal relationships
                'inhibitor',     # 15% - Blocking effects
                'mixed_strong',  # 25% - Strong parent influence
                'weak'          # 20% - Weak dependencies
            ])
            
            if cpd_pattern == 'noisy_or' and card == 2:
                # Noisy-OR: Probability increases with more active parents
                leak_prob = random.uniform(0.01, 0.15)  # Background rate
                for col in range(num_cols):
                    parent_states = []
                    temp_col = col
                    for p_card in parent_card:
                        parent_states.append(temp_col % p_card)
                        temp_col //= p_card
                    
                    # Count "active" parent states (assume last state is active)
                    active_parents = sum(1 for ps, pc in zip(parent_states, parent_card) if ps == pc - 1)
                    
                    if active_parents == 0:
                        prob_active = leak_prob
                    else:
                        # Each parent contributes independently
                        prob_per_parent = random.uniform(0.6, 0.95)
                        prob_active = 1 - (1 - leak_prob) * ((1 - prob_per_parent) ** active_parents)
                    
                    values[0, col] = 1 - prob_active
                    values[1, col] = prob_active
            
            elif cpd_pattern == 'deterministic':
                # Deterministic or near-deterministic relationships
                for col in range(num_cols):
                    parent_states = []
                    temp_col = col
                    for p_card in parent_card:
                        parent_states.append(temp_col % p_card)
                        temp_col //= p_card
                    
                    # Deterministic mapping with small noise
                    target_state = sum(parent_states) % card
                    probs = np.full(card, 0.02 / (card - 1))  # Small noise
                    probs[target_state] = 0.98
                    values[:, col] = probs
            
            elif cpd_pattern == 'inhibitor' and card == 2:
                # Inhibitor: One parent can block the effect
                base_prob = random.uniform(0.7, 0.9)
                inhibit_prob = random.uniform(0.05, 0.2)
                
                for col in range(num_cols):
                    parent_states = []
                    temp_col = col
                    for p_card in parent_card:
                        parent_states.append(temp_col % p_card)
                        temp_col //= p_card
                    
                    # If any parent is in inhibiting state (first state), low prob
                    if any(ps == 0 for ps in parent_states):
                        prob_active = inhibit_prob
                    else:
                        prob_active = base_prob
                    
                    values[0, col] = 1 - prob_active
                    values[1, col] = prob_active
            
            elif cpd_pattern == 'mixed_strong':
                # Strong influence: Parent configuration strongly determines child
                for col in range(num_cols):
                    parent_states = []
                    temp_col = col
                    for p_card in parent_card:
                        parent_states.append(temp_col % p_card)
                        temp_col //= p_card
                    
                    # Weighted sum of parent states
                    weights = np.random.uniform(0.3, 1.0, size=len(parents))
                    weighted_sum = sum(w * ps / (pc - 1) for w, ps, pc in zip(weights, parent_states, parent_card))
                    weighted_sum /= len(parents)
                    
                    # Map to probability with strong signal
                    probs = np.random.dirichlet(np.ones(card) * 0.5)
                    
                    # Bias toward state based on parent config
                    target_state = int(weighted_sum * (card - 1))
                    probs[target_state] *= random.uniform(3, 8)  # Strong bias
                    probs = probs / probs.sum()
                    
                    values[:, col] = probs
            
            else:  # weak dependencies
                # Weak influence: More randomness, less parent dependence
                for col in range(num_cols):
                    parent_states = []
                    temp_col = col
                    for p_card in parent_card:
                        parent_states.append(temp_col % p_card)
                        temp_col //= p_card
                    
                    # Base distribution
                    base_probs = np.random.dirichlet(np.ones(card) * 2)
                    
                    # Slight perturbation based on parents
                    parent_influence = sum(ps / (pc - 1) for ps, pc in zip(parent_states, parent_card)) / len(parents)
                    influence_strength = random.uniform(0.1, 0.3)
                    
                    probs = base_probs * (1 - influence_strength)
                    influenced_state = int(parent_influence * (card - 1))
                    probs[influenced_state] += influence_strength
                    
                    values[:, col] = probs
            
            cpd = TabularCPD(
                variable=node, 
                variable_card=card, 
                values=values.tolist(), 
                evidence=parents, 
                evidence_card=parent_card
            )
        
        model.add_cpds(cpd)

    assert model.check_model()
    return G, model