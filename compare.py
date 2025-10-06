# import torch
# import json
# import os

# # --- CONFIG ---
# dataset_path = "data_processing/dataset.pt"
# json_folder = "generated_graphs"
# graph_idx = 2  # Change this to any graph index you want to check
# cpd_start_idx = 9  # Update if your feature order changes
# global_cpd_len = 8 # Update if your CPD length changes

# # --- LOAD DATASET ---

# dataset = torch.load(dataset_path, weights_only=False)
# data = dataset[graph_idx]

# # --- LOAD JSON ---
# json_path = os.path.join(json_folder, f"detailed_graph_{graph_idx}.json")
# with open(json_path, "r") as f:
#     json_data = json.load(f)

# # --- NODE COMPARISON ---
# print(f"Comparing graph {graph_idx}\n")

# for node_id_str, node_json in json_data["nodes"].items():
#     node_idx = int(node_id_str)
    
#     print(f"Node {node_id_str}:")
#     print("  JSON features:", [
#         node_json["type"],
#         node_json["in_degree"],
#         node_json["out_degree"],
#         node_json["betweenness"],
#         node_json["closeness"],
#         node_json["pagerank"],
#         node_json["degree_centrality"],
#         node_json["cpd"]["variable_card"],
#         node_json["cpd"]["num_parents"]
#     ])
#     print("  PyG features:", data.x[node_idx, :cpd_start_idx].tolist())
#     print("  JSON CPD:", node_json["cpd"]["values"])
#     print("  PyG CPD:", data.x[node_idx, cpd_start_idx:cpd_start_idx+global_cpd_len].tolist())
#     print("  PyG full Json:", node_json)
#     print("PyG full features:", data.x[node_idx].tolist())
#     print()

# # --- EDGE COMPARISON ---
# print("Edge comparison:")
# for i, edge in enumerate(json_data["edges"]):
#     src = int(edge["source"])
#     tgt = int(edge["target"])
#     print(f"Edge {i}: {src} -> {tgt}")
#     print("  JSON edge attributes:", {
#         "parent_index": edge.get("parent_index", 0),
#         "parent_cardinality": edge.get("parent_cardinality", 2),
#         "is_cpd_parent": edge.get("is_cpd_parent", True)
#     })
#     # Find corresponding edge in PyG edge_index
#     for j in range(data.edge_index.shape[1]):
#         if data.edge_index[0, j] == src and data.edge_index[1, j] == tgt:
#             print("  PyG edge_attr:", data.edge_attr[j].tolist())
#             break
#     print()

import torch
import pandas as pd

# Config: Edit these as needed
dataset_path = 'datasets/train.pt'
graph_idx = 0  # Change to inspect a different graph

# Load dataset and pick a graph
dataset = torch.load(dataset_path, weights_only=False)
data = dataset[graph_idx]  # Select graph by index

# Get node features tensor
x = data.x

# Figure out feature vector length and assign proper names
static_feature_count = 10  # node_type through evidence_flag
extra_feature_names = ['cpd_entropy', 'evidence_strength', 'distance_to_evidence']
new_feature_count = len(extra_feature_names)
cpd_len = x.shape[1] - static_feature_count - new_feature_count

# Build list of all column names
feature_names = (
    [
        'node_type', 'in_degree', 'out_degree', 'betweenness', 'closeness',
        'pagerank', 'degree_centrality', 'variable_card', 'num_parents', 'evidence_flag'
    ]
    + [f'cpd_{i}' for i in range(cpd_len)]
    + extra_feature_names
)

# Convert to DataFrame
df = pd.DataFrame(x.numpy(), columns=feature_names)

# Print graph summary
print(f"\n--- Node Feature Table for Graph {graph_idx} ---")
print(f"Shape: {df.shape[0]} nodes x {df.shape[1]} features")
print(f"Features: {list(df.columns)}\n")

# Show all features and values for all nodes, without truncation:
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 250):
    print(df)

# If you want only to display the DataFrame summary and not all rows (for huge graphs), comment out the 'with ... print(df)' block and just use 'print(df.head(20))'
