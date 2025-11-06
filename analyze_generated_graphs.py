# import os
# import json
# import matplotlib.pyplot as plt
# import networkx as nx
# from tqdm import tqdm
# import numpy as np
# from collections import Counter

# # ========= Configuration =========
# FOLDER = os.path.join(os.path.dirname(__file__), "generated_graphs")  # path to folder
# NUM_SAMPLE_VIS = 6  # number of graphs to visualize
# SAVE_PLOTS = True   # if you want to save the plots as images
# # =================================

# def load_graph(file_path):
#     with open(file_path, "r") as f:
#         data = json.load(f)
#     return data

# def analyze_graph(graph_json):
#     metadata = graph_json.get("metadata", {})
#     nodes = graph_json.get("nodes", {})
#     edges = graph_json.get("edges", [])
#     node_types = graph_json.get("node_types", {})

#     total_nodes = metadata.get("total_nodes", len(nodes))
#     total_edges = metadata.get("total_edges", len(edges))
#     has_cycles = metadata.get("has_cycles", False)
#     max_path_len = graph_json.get("paths_info", {}).get("max_path_length", -1)

#     # Count node types
#     type_counts = Counter([n.get("type", "unknown") for n in nodes.values()])

#     # Gather CPD statistics
#     variable_cards = []
#     num_parents = []
#     cpd_lengths = []
#     for n in nodes.values():
#         cpd = n.get("cpd", {})
#         variable_cards.append(cpd.get("variable_card", 0))
#         num_parents.append(cpd.get("num_parents", 0))
#         cpd_lengths.append(len(cpd.get("values", [])))

#     return {
#         "nodes": total_nodes,
#         "edges": total_edges,
#         "roots": len(node_types.get("roots", [])),
#         "leaves": len(node_types.get("leaves", [])),
#         "intermediates": type_counts.get("intermediate", 0),
#         "has_cycles": has_cycles,
#         "max_path_len": max_path_len,
#         "variable_cards": variable_cards,
#         "num_parents": num_parents,
#         "cpd_lengths": cpd_lengths,
#     }

# def visualize_graph(graph_json, ax):
#     G = nx.DiGraph()
#     nodes = graph_json.get("nodes", {})
#     edges = graph_json.get("edges", [])

#     for node_id in nodes.keys():
#         G.add_node(int(node_id))

#     for e in edges:
#         G.add_edge(int(e["source"]), int(e["target"]))

#     pos = nx.spring_layout(G, seed=42)
#     nx.draw(
#         G, pos, node_size=20, arrowsize=5, alpha=0.7, with_labels=False, ax=ax
#     )
#     ax.set_title(f"Graph #{graph_json.get('graph_index', '?')} | "
#                  f"Nodes: {len(nodes)}, Edges: {len(edges)}")

# def main():
#     all_graphs = [
#         os.path.join(FOLDER, f)
#         for f in os.listdir(FOLDER)
#         if f.endswith(".json") and "detailed_graph_" in f
#     ]
#     all_graphs.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))

#     if not all_graphs:
#         print("❌ No JSON graph files found in:", FOLDER)
#         return

#     print(f"📂 Found {len(all_graphs)} graph files. Analyzing...\n")

#     stats = []
#     for fpath in tqdm(all_graphs, desc="Processing graphs"):
#         try:
#             gjson = load_graph(fpath)
#             stats.append(analyze_graph(gjson))
#         except Exception as e:
#             print(f"⚠️ Error in {fpath}: {e}")

#     # Convert stats list to arrays
#     num_nodes = np.array([s["nodes"] for s in stats])
#     num_edges = np.array([s["edges"] for s in stats])
#     max_paths = np.array([s["max_path_len"] for s in stats])

#     # === Summary ===
#     print("\n📊 SUMMARY STATISTICS:")
#     print(f"Graphs analyzed: {len(stats)}")
#     print(f"Nodes: mean={num_nodes.mean():.1f}, min={num_nodes.min()}, max={num_nodes.max()}")
#     print(f"Edges: mean={num_edges.mean():.1f}, min={num_edges.min()}, max={num_edges.max()}")
#     print(f"Max path length: mean={max_paths.mean():.1f}")
#     print(f"Roots avg: {np.mean([s['roots'] for s in stats]):.1f}")
#     print(f"Leaves avg: {np.mean([s['leaves'] for s in stats]):.1f}")
#     print(f"Intermediate avg: {np.mean([s['intermediates'] for s in stats]):.1f}")

#     # === Plots ===
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.hist(num_nodes, bins=30)
#     plt.title("Distribution of Node Counts")
#     plt.xlabel("Nodes")
#     plt.ylabel("Frequency")

#     plt.subplot(1, 2, 2)
#     plt.hist(num_edges, bins=30)
#     plt.title("Distribution of Edge Counts")
#     plt.xlabel("Edges")
#     plt.ylabel("Frequency")

#     plt.tight_layout()
#     if SAVE_PLOTS:
#         plt.savefig("graph_statistics.png", dpi=300)
#     plt.show()

#     # === Visualize sample graphs ===
#     sample_graphs = np.random.choice(all_graphs, size=min(NUM_SAMPLE_VIS, len(all_graphs)), replace=False)
#     fig, axes = plt.subplots(1, len(sample_graphs), figsize=(5*len(sample_graphs), 4))
#     if len(sample_graphs) == 1:
#         axes = [axes]

#     for fpath, ax in zip(sample_graphs, axes):
#         gjson = load_graph(fpath)
#         visualize_graph(gjson, ax)

#     plt.tight_layout()
#     if SAVE_PLOTS:
#         plt.savefig("sample_graphs.png", dpi=300)
#     plt.show()

#     # === CPD statistics ===
#     all_cpd_lengths = [x for s in stats for x in s["cpd_lengths"]]
#     plt.figure(figsize=(6,4))
#     plt.hist(all_cpd_lengths, bins=30)
#     plt.title("Distribution of CPD Lengths")
#     plt.xlabel("Flattened CPD Length")
#     plt.ylabel("Frequency")
#     plt.tight_layout()
#     if SAVE_PLOTS:
#         plt.savefig("cpd_length_distribution.png", dpi=300)
#     plt.show()

# if __name__ == "__main__":
#     main()


import os
import json
from collections import Counter

json_folder = "generated_graphs"  # your folder
json_files = [f for f in os.listdir(json_folder) if f.startswith("detailed_graph_") and f.endswith(".json")]

node_counts = []
edge_counts = []
root_counts = []
intermediate_counts = []
leaf_counts = []

for f in json_files:
    path = os.path.join(json_folder, f)
    with open(path, "r") as jf:
        data = json.load(jf)
        
    num_nodes = len(data["nodes"])
    num_edges = len(data["edges"])
    roots = len(data["node_types"].get("roots", []))
    intermediates = len(data["node_types"].get("intermediates", []))
    leaves = len(data["node_types"].get("leaves", []))
    
    node_counts.append(num_nodes)
    edge_counts.append(num_edges)
    root_counts.append(roots)
    intermediate_counts.append(intermediates)
    leaf_counts.append(leaves)

# Summary statistics
print("=== Node counts across graphs ===")
for k, v in sorted(Counter(node_counts).items()):
    print(f"{k} nodes: {v} graphs")

print("\n=== Edge counts across graphs ===")
for k, v in sorted(Counter(edge_counts).items()):
    print(f"{k} edges: {v} graphs")

print("\n=== Root nodes per graph ===")
for k, v in sorted(Counter(root_counts).items()):
    print(f"{k} roots: {v} graphs")

print("\n=== Intermediate nodes per graph ===")
for k, v in sorted(Counter(intermediate_counts).items()):
    print(f"{k} intermediates: {v} graphs")

print("\n=== Leaf nodes per graph ===")
for k, v in sorted(Counter(leaf_counts).items()):
    print(f"{k} leaves: {v} graphs")

