"""
Bayesian Network to PyG Dataset Converter

Converts Bayesian network structures (NetworkX graphs) to PyTorch Geometric
Data objects for GNN training. Handles node features, edge indices, and
edge attributes with optional padding for CPD alignment.

Main Functions:
    - from_networkx: Convert NetworkX DiGraph to PyG Data format
    - load_bn_graphs: Load and convert all BN graphs from a directory
"""

# bn_nx_to_pyg_dataset.py
import os
import json
import torch
import networkx as nx
from torch_geometric.data import Data

def from_networkx(G: nx.DiGraph, use_edge_attr=True):
    # Node feature matrix
    #x = torch.stack([G.nodes[n]["x"] for n in G.nodes()])

    sorted_nodes = sorted(G.nodes(), key=int)
    x = torch.stack([G.nodes[n]["x"] for n in sorted_nodes])
    # Edge list
    edge_index = torch.tensor(list(G.edges())).t().contiguous()

    if use_edge_attr:
        edge_attr = torch.stack([
            torch.tensor([
                G.edges[edge].get("parent_index", 0),
                G.edges[edge].get("parent_cardinality", 2),
                float(G.edges[edge].get("is_cpd_parent", True))
            ], dtype=torch.float)
            for edge in G.edges()
        ])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return Data(x=x, edge_index=edge_index)

def load_bn_graphs(folder_path, global_cpd_len, use_edge_attr=True):
    data_list = []
    padding_count = 0
    file_count = 0
    filenames = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'detailed_graph_' in x else 0)
    for filename in filenames:
    #for filename in os.listdir(folder_path):
        if filename.endswith(".json") and "graph" in filename:
            file_count += 1
            with open(os.path.join(folder_path, filename)) as f:
                graph_json = json.load(f)
            nx_graph = nx.DiGraph()
            
            graph_metadata = graph_json.get("metadata", {})
            nodes = graph_json["nodes"]
            edges = graph_json["edges"]

            for node_id_str, attr in nodes.items():
                node_id = int(node_id_str)
                node_type_map = {"root": 0, "intermediate": 1, "leaf": 2}
                node_type = node_type_map.get(attr["type"], -1)

                # Basic node features
                node_features = [
                    node_type,
                    attr["in_degree"],
                    attr["out_degree"],
                    attr["betweenness"],
                    attr["closeness"],
                    attr["pagerank"],
                    attr["degree_centrality"]
                ]

                # CPD-related features
                cpd_info = attr["cpd"]
                variable_card = cpd_info.get("variable_card", 2)
                num_parents = cpd_info.get("num_parents", 0)
                cpd_values = cpd_info["values"]

                # Verification and padding of CPD values
                if len(cpd_values) != global_cpd_len:
                    print(f" CPD length mismatch in {filename}, node {node_id}: {len(cpd_values)} != {global_cpd_len}")
                    padding_count += 1
                    if len(cpd_values) < global_cpd_len:
                        cpd_values += [0.0] * (global_cpd_len - len(cpd_values))
                    else:
                        cpd_values = cpd_values[:global_cpd_len]

                node_features.extend([variable_card, num_parents])
                node_features.extend(cpd_values)

                # Add node with features
                nx_graph.add_node(
                    node_id,
                    #x=torch.tensor(node_features, dtype=torch.float),
                    x=torch.tensor(node_features.copy(), dtype=torch.float),
                    evidence=cpd_info.get("evidence", []),
                    evidence_card=cpd_info.get("evidence_card", [])
                )

            for edge in edges:
                nx_graph.add_edge(
                    int(edge["source"]),
                    int(edge["target"]),
                    parent_index=edge.get("parent_index", 0),
                    parent_cardinality=edge.get("parent_cardinality", 2),
                    is_cpd_parent=edge.get("is_cpd_parent", True)
                )

            # Convert to PyG Data object
            pyg_graph = from_networkx(nx_graph, use_edge_attr=use_edge_attr)
            pyg_graph.name = filename.replace(".json", "") 

            # Add graph-level metadata
            pyg_graph.max_path_length = graph_json.get("paths_info", {}).get("max_path_length", -1)
            pyg_graph.num_nodes = graph_metadata.get("total_nodes", nx_graph.number_of_nodes())
            pyg_graph.num_edges = graph_metadata.get("total_edges", nx_graph.number_of_edges())
            pyg_graph.has_cycles = graph_metadata.get("has_cycles", False)

            data_list.append(pyg_graph)

    print(f"\n Processed {file_count} graphs. {padding_count} nodes required CPD length correction.\n")
    return data_list
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpd_len_path = os.path.join(script_dir, "..", "global_datasets", "global_cpd_len.txt")

    with open(cpd_len_path, "r") as f:
        global_cpd_len = int(f.read().strip())   
    #with open("../global_datasets/global_cpd_len.txt", "r") as f:
     #   global_cpd_len = int(f.read().strip())

    #folder = "../generated_graphs" 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, "..", "generated_graphs")
    folder = os.path.abspath(folder)  # Ensure absolute path

    use_edge_attr = True

    dataset = load_bn_graphs(folder, global_cpd_len, use_edge_attr)
    torch.save(dataset, "dataset.pt")
    print(f" Dataset saved. \n Global CPD length was {global_cpd_len}")

