import os
import json
import torch
import networkx as nx
from torch_geometric.data import Data

def from_networkx(G: nx.DiGraph):
    x = torch.stack([G.nodes[n]["x"] for n in G.nodes()])
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    return Data(x=x, edge_index=edge_index)

def load_bn_graphs(folder_path, global_cpd_len):
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and "graph" in filename:
            with open(os.path.join(folder_path, filename)) as f:
                graph_json = json.load(f)
            nx_graph = nx.DiGraph()
            for node_id_str, attr in graph_json["nodes"].items():
                node_id = int(node_id_str)
                
                # Determine node type based on degree
                in_deg = attr["in_degree"]
                out_deg = attr["out_degree"]
                if in_deg == 0:
                    node_type = 0  # root
                elif out_deg == 0:
                    node_type = 2  # leaf
                else:
                    node_type = 1  # intermediate
                node_features = [node_type]
                node_features.extend([
                    in_deg,
                    out_deg,
                    attr["betweenness"],
                    attr["closeness"],
                    attr["pagerank"],
                    attr["degree_centrality"]
                ])
                
                # Get CPD values (flatten if nested)
                cpd_values = attr["cpd"]["values"]
                if any(isinstance(v, list) for v in cpd_values):
                    cpd_values = [item for sublist in cpd_values for item in sublist]
                
                # Pad or truncate CPD to global length
                if len(cpd_values) < global_cpd_len:
                    cpd_values += [0.0] * (global_cpd_len - len(cpd_values))
                else:
                    cpd_values = cpd_values[:global_cpd_len]
                
                # Append CPD values at the end
                node_features.extend(cpd_values)
                
                # Convert to tensor and add to graph
                nx_graph.add_node(node_id, x=torch.tensor(node_features, dtype=torch.float))
            
            # Add edges
            nx_graph.add_edges_from((int(u), int(v)) for u, v in graph_json["edges"])
            
            # Convert to PyG Data object
            pyg_graph = from_networkx(nx_graph)
            data_list.append(pyg_graph)
    return data_list
