import os
import json
import networkx as nx
import pandas as pd
import pickle
import numpy as np
import itertools
from networkx.readwrite import json_graph

def save_graph(G, config, index):
    path = os.path.join(config['output_dir'], f'graph_{index}.json')
    data = nx.node_link_data(G, edges="links") 
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_cpds_as_numpy_tables(model, filename):
    cpd_tables = {}

    for cpd in model.get_cpds():
        variable = cpd.variable
        var_card = cpd.variable_card

        evidence = cpd.get_evidence()
        evidence_card = cpd.cardinality[1:] 

        if evidence:
            parent_combos = list(itertools.product(*[range(card) for card in evidence_card]))

            rows = []
            cpd_reshaped = cpd.values.reshape(var_card, -1)
            for i, combo in enumerate(parent_combos):
                for state in range(var_card):
                    row = list(combo) + [state, cpd_reshaped[state, i]]
                    rows.append(row)

            columns = evidence + [f"{variable}_state", "P"]
            df = pd.DataFrame(rows, columns=columns)
        else:
            df = pd.DataFrame({
                f"{variable}_state": list(range(var_card)),
                "P": cpd.values.flatten()
            })

        cpd_tables[variable] = df

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, cpd_tables)

# Computes maximum CPD vector length for one model
def get_max_cpd_length(model):  
    max_len = 0
    for cpd in model.get_cpds():
        values = cpd.values.tolist()
        if isinstance(values[0], list):
            flat = [v for row in values for v in row]
        else:
            flat = values
            max_len = max(max_len, len(flat))
            return max_len

# Computes global max across all models
def compute_global_max_cpd_length(models): 
    max_len = 0
    for model in models:
        for cpd in model.get_cpds():
            flat = flatten(cpd.values.tolist())
            max_len = max(max_len, len(flat))
    return max_len

# Create mapping from flattened position to parent configuration
def create_cpd_position_map(cpd_values_shape, evidence_card):
    if not evidence_card:
        return [{"parent_config": [], "child_state": i, "flattened_position": i} 
                for i in range(cpd_values_shape[0])]
    
    parent_configs = list(itertools.product(*[range(card) for card in evidence_card]))
    position_map = []
    
    pos = 0
   
    for parent_config in parent_configs:
        for child_state in range(cpd_values_shape[0]):
            position_map.append({
                "parent_config": list(parent_config),
                "child_state": child_state,
                "flattened_position": pos
            })
            pos += 1
    
    return position_map

def save_graph_with_details(G, model, config, index, global_max_cpd_len):
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)
    degree_centrality = nx.degree_centrality(G)
    max_cpd_len = global_max_cpd_len

    cpds = {}
    cpd_lookup = {} 
    
    for cpd in model.get_cpds():
        evidence = cpd.get_evidence()
        evidence_card = [int(model.get_cardinality()[ev]) for ev in evidence] if evidence else []
        variable = cpd.variable
        padded_values, validity_mask = pad_cpd_values_fixed(cpd.values.tolist(), target_len=max_cpd_len, 
                                                           evidence_card=evidence_card, 
                                                           variable_card=cpd.variable_card)
        
        cpd_data = {
            "values": pad_cpd_values(cpd.values.tolist(), target_len=max_cpd_len),
            "evidence": evidence,
            "evidence_card": evidence_card,
            "original_shape": list(cpd.values.shape),
            "num_parents": len(evidence),
            "variable_card": cpd.variable_card,
            "position_map": create_cpd_position_map(cpd.values.shape, evidence_card)
        }
        
        cpds[variable] = cpd_data
        cpd_lookup[variable] = cpd_data

    node_data = {}
    for node in G.nodes:
        node_data[str(node)] = {
            "type": G.nodes[node].get("type", ""),
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
            "betweenness": betweenness.get(node, 0),
            "closeness": closeness.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "degree_centrality": degree_centrality.get(node, 0),
            "cpd": cpds.get(node, {})
        }

    edge_list = []
    for parent, child in G.edges():
        # Get the child's CPD information
        child_cpd = cpd_lookup.get(child, {})
        parent_evidence = child_cpd.get("evidence", [])
        parent_evidence_card = child_cpd.get("evidence_card", [])
        
        # Find this parent's index in the child's evidence list
        parent_index = -1
        parent_cardinality = 2  # default
        if parent in parent_evidence:
            parent_index = parent_evidence.index(parent)
            if parent_index < len(parent_evidence_card):
                parent_cardinality = parent_evidence_card[parent_index]
        
        edge_info = {
            "source": parent,
            "target": child,
            "parent_index": parent_index,
            "parent_cardinality": parent_cardinality,
            "is_cpd_parent": parent in parent_evidence
        }
        edge_list.append(edge_info)

    # Get node type lists for query information
    roots = [n for n in G.nodes() if G.nodes[n]['type'] == 'root']
    leaves = [n for n in G.nodes() if G.nodes[n]['type'] == 'leaf']
    intermediates = [n for n in G.nodes() if G.nodes[n]['type'] == 'intermediate']
    
    # Calculate maximum path length for inference queries
    max_path_length = 0
    if roots and leaves:
        try:
            path_lengths = []
            for root in roots:
                for leaf in leaves:
                    if nx.has_path(G, root, leaf):
                        path_lengths.append(len(nx.shortest_path(G, root, leaf)) - 1)
            max_path_length = max(path_lengths) if path_lengths else 0
        except:
            max_path_length = 0

    full_graph_data = {
        "graph_index": index,
        "nodes": node_data,
        "edges": edge_list,
        "node_types": {  
        "roots": [n for n in G.nodes() if G.nodes[n]['type'] == 'root'],
        "leaves": [n for n in G.nodes() if G.nodes[n]['type'] == 'leaf'],
        "intermediates": [n for n in G.nodes() if G.nodes[n]['type'] == 'intermediate']
    },
    "paths_info": {  
            "max_path_length": max_path_length
        },
        "metadata": {
            "global_max_cpd_length": max_cpd_len,
            "total_nodes": len(G.nodes()),
            "total_edges": len(G.edges()),
            "has_cycles": not nx.is_directed_acyclic_graph(G) if G.is_directed() else False
        }
    }

    os.makedirs(config['output_dir'], exist_ok=True)
    out_path = os.path.join(config['output_dir'], f"detailed_graph_{index}.json")
    with open(out_path, 'w') as f:
        json.dump(full_graph_data, f, indent=2)

# Flatten nested lists.
def flatten(values):
    if isinstance(values, list):
        result = []
        for v in values:
            result.extend(flatten(v))
        return result
    else:
        return [values]

def pad_cpd_values(values, target_len):
     values = flatten(values)
     if len(values) < target_len:
         values += [0.0] * (target_len - len(values))
     else:
         values = values[:target_len]
     return values

# Padding function
def pad_cpd_values_fixed(values, target_len, evidence_card, variable_card):
   
    flat_values = flatten(values)
    validity_mask = [1] * len(flat_values)  
    
    if len(flat_values) < target_len:
        
        if evidence_card:
            
            num_parent_configs = np.prod(evidence_card)
            prob_per_child_state = 1.0 / variable_card
            
            padding_needed = target_len - len(flat_values)
            padded_values = flat_values + [prob_per_child_state] * padding_needed
            validity_mask = validity_mask + [0] * padding_needed  
        else:
            
            padded_values = flat_values + [0.0] * (target_len - len(flat_values))
            validity_mask = validity_mask + [0] * (target_len - len(flat_values))
    else:
        
        padded_values = flat_values[:target_len]
        validity_mask = validity_mask[:target_len]
    
    return padded_values, validity_mask