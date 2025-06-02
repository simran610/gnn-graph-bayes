# File: exporter.py
import os
import json
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import networkx as nx
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

def get_max_cpd_length(model):
    max_len = 0
    for cpd in model.get_cpds():
        values = cpd.values.tolist()
        if isinstance(values[0], list):
            flat = [v for row in values for v in row]
        else:
            flat = values  # Already flat
        max_len = max(max_len, len(flat))
    return max_len

def save_graph_with_details(G, model, config, index):
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)
    degree_centrality = nx.degree_centrality(G)

    # Get max CPD length for current model
    max_cpd_len = get_max_cpd_length(model)

    # CPD data: flatten and pad
    cpds = {
        cpd.variable: {
            "values": pad_cpd_values(cpd.values.tolist(), target_len=max_cpd_len),
            "evidence": cpd.get_evidence(),
            "evidence_card": [
                int(model.get_cardinality()[ev]) for ev in cpd.get_evidence()
            ]
        } for cpd in model.get_cpds()
    }

    # Combine node info
    node_data = {}
    for node in G.nodes:
        node_data[str(node)] = {
            "type": G.nodes[node].get("type", ""),
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
        #   "eigenvector": eigen.get(node, 0),
            "betweenness": betweenness.get(node, 0),
            "closeness": closeness.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "degree_centrality": degree_centrality.get(node, 0),
            "cpd": cpds.get(node, {})
        }

    # Edges
    edge_list = list(G.edges())

    # Combine everything
    full_graph_data = {
        "graph_index": index,
        "nodes": node_data,
        "edges": edge_list
    }

    # Save
    os.makedirs(config['output_dir'], exist_ok=True)
    out_path = os.path.join(config['output_dir'], f"detailed_graph_{index}.json")
    with open(out_path, 'w') as f:
        json.dump(full_graph_data, f, indent=2)

def pad_cpd_values(values, target_len):
    if isinstance(values[0], list):
        flat = [v for row in values for v in row]
    else:
        flat = values  # Already flat
    padded = flat + [0.0] * (target_len - len(flat))
    return padded
