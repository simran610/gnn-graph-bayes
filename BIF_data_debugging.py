# import os
# import json
# import random
# import torch
# import yaml
# import numpy as np
# import scipy.stats
# import networkx as nx
# import glob
# from pathlib import Path
# from typing import Dict, List
# from itertools import combinations, product

# from pgmpy.readwrite import BIFReader
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.models import BayesianNetwork
# from pgmpy.inference import VariableElimination
# from torch_geometric.data import Data


# class BIFDownloader:
#     BNLEARN_REPO = "https://www.bnlearn.com/bnrepository"
#     EXAMPLE_NETWORKS = [
#         "asia", "cancer", "child", "insurance", "alarm",
#         "barley", "hailfinder", "hepar2", "win95pts"
#     ]

#     def __init__(self, output_dir="dataset_bif_files", verbose=False):
#         self.output_dir = output_dir
#         self.verbose = verbose
#         os.makedirs(output_dir, exist_ok=True)

#     def download_bif(self, network_name: str) -> str:
#         bif_url = f"{self.BNLEARN_REPO}/{network_name}/{network_name}.bif"
#         local_path = os.path.join(self.output_dir, f"{network_name}.bif")
#         if os.path.exists(local_path):
#             if self.verbose:
#                 print(f"✓ {network_name}.bif already exists")
#             return local_path
#         try:
#             if self.verbose:
#                 print(f"Downloading {network_name}.bif...")
#             from urllib.request import urlretrieve
#             urlretrieve(bif_url, local_path)
#             if self.verbose:
#                 print(f"✓ Saved to {local_path}")
#             return local_path
#         except Exception as e:
#             print(f"✗ Failed to download {network_name}: {e}")
#             return None

#     def download_multiple(self, networks=None):
#         if networks is None:
#             networks = self.EXAMPLE_NETWORKS[:3]
#         paths = []
#         for net in networks:
#             path = self.download_bif(net)
#             if path:
#                 paths.append(path)
#         return paths

# class BIFToGraphConverter:
#     def __init__(self, verbose=False):
#         self.verbose = verbose

#     @staticmethod
#     def convert_np_types(obj):
#         if isinstance(obj, dict):
#             return {k: BIFToGraphConverter.convert_np_types(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [BIFToGraphConverter.convert_np_types(i) for i in obj]
#         elif isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.bool_):
#             return bool(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return obj

#     def load_bif(self, bif_path: str):
#         try:
#             reader = BIFReader(bif_path)
#             model = reader.get_model()
#             metadata = {
#                 "nodes": list(model.nodes()),
#                 "edges": list(model.edges()),
#                 "num_nodes": model.number_of_nodes(),
#                 "num_edges": model.number_of_edges()
#             }
#             if self.verbose:
#                 print(f"✓ Loaded BIF: {metadata['num_nodes']} nodes, {metadata['num_edges']} edges")
#             return model, metadata
#         except Exception as e:
#             print(f"✗ Error loading BIF: {e}")
#             raise

#     def get_node_types(self, model: BayesianNetwork):
#         roots, intermediates, leaves = [], [], []
#         for node in model.nodes():
#             parents = list(model.predecessors(node))
#             children = list(model.successors(node))
#             if len(parents) == 0 and len(children) > 0:
#                 roots.append(node)
#             elif len(parents) > 0 and len(children) > 0:
#                 intermediates.append(node)
#             else:
#                 leaves.append(node)
#         return {"roots": roots, "intermediates": intermediates, "leaves": leaves}

#     def extract_cpd_as_dict(self, model, node: str):
#         cpd = model.get_cpds(node)
#         if cpd is None:
#             return {"variable_card": 2, "evidence": [], "evidence_card": [], "values": [0.5, 0.5]}
#         return {
#             "variable_card": cpd.variable_card,
#             "evidence": list(cpd.variables[1:]) if len(cpd.variables) > 1 else [],
#             "evidence_card": list(cpd.cardinality[1:]) if len(cpd.cardinality) > 1 else [],
#             "values": cpd.values.flatten().tolist()
#         }

#     def convert_to_json(self, model, bif_name: str, output_dir: str = "bif_json"):
#         os.makedirs(output_dir, exist_ok=True)
#         node_types = self.get_node_types(model)
#         nodes_dict = {}
#         for node in model.nodes():
#             cpd = self.extract_cpd_as_dict(model, node)
#             nodes_dict[str(node)] = {
#                 "cpd": cpd,
#                 "node_type": self._get_node_type_label(node, node_types)
#             }
#         edges = [{"source": str(src), "target": str(tgt)} for src, tgt in model.edges()]
#         json_data = {
#             "network_name": bif_name,
#             "nodes": nodes_dict,
#             "edges": edges,
#             "node_types": {
#                 "roots": [str(n) for n in node_types["roots"]],
#                 "intermediates": [str(n) for n in node_types["intermediates"]],
#                 "leaves": [str(n) for n in node_types["leaves"]],
#             },
#         }
#         json_data = BIFToGraphConverter.convert_np_types(json_data)
#         json_path = os.path.join(output_dir, f"{bif_name}.json")
#         with open(json_path, "w") as f:
#             json.dump(json_data, f, indent=2)
#         if self.verbose:
#             print(f"✓ Saved JSON: {json_path}")
#         return json_path

#     def _get_node_type_label(self, node: str, node_types: Dict):
#         if node in node_types["roots"]:
#             return 0
#         elif node in node_types["intermediates"]:
#             return 1
#         else:
#             return 2

# class GraphFeatureExtractor:
#     def __init__(self, verbose=False):
#         self.verbose = verbose

#     def compute_structural_features(self, G: nx.DiGraph, node: str):
#         in_deg = G.in_degree[node]
#         out_deg = G.out_degree[node]
#         try:
#             betweenness = nx.betweenness_centrality(G).get(node, 0.0)
#             closeness = nx.closeness_centrality(G).get(node, 0.0)
#             pagerank = nx.pagerank(G).get(node, 0.0)
#             degree_cent = nx.degree_centrality(G).get(node, 0.0)
#         except Exception:
#             betweenness = closeness = pagerank = degree_cent = 0.0
#         return [float(in_deg), float(out_deg), float(betweenness), float(closeness), float(pagerank), float(degree_cent)]
    
#     # ====== ADDED: Reduce CPD feature dimensionality by summary statistics ======
#     def extract_cpd_summary_feats(self, cpd_values):
#         cpd_arr = np.array(cpd_values)
#         feats = [
#             np.mean(cpd_arr),
#             np.std(cpd_arr),
#             np.min(cpd_arr),
#             np.max(cpd_arr),
#             scipy.stats.entropy(cpd_arr) if (cpd_arr>0).any() else 0.0,
#             np.argmax(cpd_arr),
#             np.count_nonzero(cpd_arr),
#             np.median(cpd_arr),
#             np.percentile(cpd_arr, 25),
#             np.percentile(cpd_arr, 75),
#         ]
#         return feats
#     # ====== END ADDED ======

#     def extract_node_features(self, model, json_data, G, node, node_type: int, global_cpd_len: int):
#         cpd_info = json_data["nodes"][node]["cpd"]
#         variable_card = cpd_info["variable_card"]
#         num_parents = len(cpd_info["evidence"])
#         struct_feat = self.compute_structural_features(G, node)
#         cpd_values = cpd_info["values"]
#         # ====== ORIGINAL (if you want massive features, keep this enabled) ======
#         # if len(cpd_values) < global_cpd_len:
#         #     cpd_values = cpd_values + [0.0] * (global_cpd_len - len(cpd_values))
#         # else:
#         #     cpd_values = cpd_values[:global_cpd_len]
#         # features = [float(node_type)] + struct_feat + [float(variable_card), float(num_parents), 0.0] + cpd_values

#         # ====== REPLACE with following line to use CPD summary statistics: ======
#         cpd_feats = self.extract_cpd_summary_feats(cpd_values)
#         features = [float(node_type)] + struct_feat + [float(variable_card), float(num_parents), 0.0] + cpd_feats

#         return np.array(features, dtype=np.float32)
#     # ====== END of MODIFICATIONS ======

#     def create_graph_dataset(self, model, json_data, global_cpd_len):
#         G = nx.DiGraph()
#         nodes = list(json_data["nodes"].keys())
#         G.add_nodes_from(nodes)
#         for edge in json_data["edges"]:
#             G.add_edge(edge["source"], edge["target"])
#         node_to_idx = {n: i for i, n in enumerate(sorted(nodes))}
#         node_features = [
#             self.extract_node_features(model, json_data, G, node,
#                                        json_data["nodes"][node]["node_type"], global_cpd_len)
#             for node in sorted(nodes)
#         ]
#         x = torch.tensor(np.array(node_features), dtype=torch.float32)
#         edge_list = [[node_to_idx[edge["source"]], node_to_idx[edge["target"]]] for edge in json_data["edges"]]
#         edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
#         data = Data(x=x, edge_index=edge_index, y=torch.tensor([0.5, 0.5], dtype=torch.float32))
#         return data, node_to_idx


# class EnhancedGraphPreprocessor:
#     def __init__(self, verbose=False):
#         self.verbose = verbose

#     def compute_cpd_entropy(self, cpd_values):
#         p = cpd_values / np.sum(cpd_values) if np.sum(cpd_values) > 0 else np.ones_like(cpd_values) / len(cpd_values)
#         return 0. if np.allclose(p, 0) else scipy.stats.entropy(p)

#     def compute_distance_to_evidence(self, edge_index, evidence_indices, num_nodes):
#         if len(evidence_indices) == 0:
#             return np.full(num_nodes, num_nodes, dtype=np.float32)
#         G = nx.Graph()
#         G.add_edges_from(edge_index.t().cpu().numpy())
#         dists = []
#         for i in range(num_nodes):
#             min_dist = num_nodes
#             for e in evidence_indices:
#                 try:
#                     if nx.has_path(G, i, int(e)):
#                         dist = nx.shortest_path_length(G, i, int(e))
#                         if dist < min_dist:
#                             min_dist = dist
#                 except:
#                     pass
#             dists.append(min_dist)
#         return np.array(dists, dtype=np.float32)

#     def add_missing_features(self, data, global_cpd_len, graph_idx=None):
#         x = data.x.clone().float()
#         e_flag = 9
#         cpd_start = 10
#         cpd_end = cpd_start + global_cpd_len
#         num_nodes = x.size(0)
#         entropies = [self.compute_cpd_entropy(x[i, cpd_start:cpd_end].cpu().numpy()) for i in range(num_nodes)]
#         cpd_entropy = torch.tensor(entropies, dtype=torch.float32).unsqueeze(1)
#         evidence_str = x[:, e_flag:e_flag + 1].clone()
#         evidence_indices = (x[:, e_flag] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
#         distances = torch.tensor(self.compute_distance_to_evidence(data.edge_index, evidence_indices, num_nodes),
#                                  dtype=torch.float32).unsqueeze(1)
#         x_enh = torch.cat([x, cpd_entropy, evidence_str, distances], dim=1)
#         data.x = x_enh
#         if self.verbose and graph_idx is not None:
#             print(f"  Graph {graph_idx}: Features {x.shape[1]} → {x_enh.shape[1]}")
#         return data


# class AutoInferenceEngine:
#     """Auto-generate evidence and run inference on BIF models"""
    
#     def __init__(self, max_evidence_nodes=3, max_evidence_combinations=5, verbose=False):
#         self.max_evidence_nodes = max_evidence_nodes
#         self.max_evidence_combinations = max_evidence_combinations
#         self.verbose = verbose
    
#     def _get_node_states(self, model, node):
#         """Get all possible state names for a node"""
#         for cpd in model.get_cpds():
#             if cpd.variable == node:
#                 return cpd.state_names[node]
#         return [0, 1]  # fallback
    
#     def _generate_evidence_combinations(self, model, json_data):
#         """Generate multiple evidence combinations from non-root nodes"""
#         root_nodes = set(json_data.get("node_types", {}).get("roots", []))
#         non_root_nodes = [n for n in json_data["nodes"].keys() if n not in root_nodes]
        
#         if not non_root_nodes:
#             return [{}]
        
#         evidence_list = []
        
#         # Single node evidence
#         for node in non_root_nodes[:self.max_evidence_nodes]:
#             states = self._get_node_states(model, node)
#             for state in states:
#                 evidence_list.append({node: state})
        
#         # Multi-node evidence (combinations)
#         for r in range(2, min(self.max_evidence_nodes + 1, len(non_root_nodes) + 1)):
#             for node_combo in combinations(non_root_nodes, r):
#                 state_combos = []
#                 for node in node_combo:
#                     states = self._get_node_states(model, node)
#                     state_combos.append([(node, s) for s in states])
                
#                 for state_tuple in product(*state_combos):
#                     evidence_list.append(dict(state_tuple))
#                     if len(evidence_list) >= self.max_evidence_combinations:
#                         return evidence_list
        
#         return evidence_list[:self.max_evidence_combinations]
    
#     def run_inference(self, model, json_data):
#         """Run inference and return probabilities"""
#         try:
#             inference = VariableElimination(model)
#             root_nodes = json_data.get("node_types", {}).get("roots", [])
            
#             if not root_nodes:
#                 return None, None
            
#             root_node = str(root_nodes[0])
            
#             # Generate evidence combinations
#             evidence_combinations = self._generate_evidence_combinations(model, json_data)
            
#             results = []
#             for evidence in evidence_combinations:
#                 try:
#                     if not evidence:  # No evidence case
#                         query_result = inference.query(variables=[root_node])
#                     else:
#                         query_result = inference.query(
#                             variables=[root_node],
#                             evidence=evidence,
#                             show_progress=False
#                         )
                    
#                     probs = query_result.values
#                     states = query_result.state_names[root_node]
                    
#                     prob_dict = {str(state): float(probs[i]) for i, state in enumerate(states)}
                    
#                     results.append({
#                         'root_node': root_node,
#                         'root_node_probs': prob_dict,
#                         'evidence': {str(k): str(v) for k, v in evidence.items()},
#                     })
                    
#                 except Exception as e:
#                     if self.verbose:
#                         print(f"  Warning: Single inference failed: {e}")
#                     continue
            
#             if results:
#                 # Return first result as primary inference target
#                 primary = results[0]
#                 return primary['root_node_probs'], primary['evidence']
#             else:
#                 return None, None
                
#         except Exception as e:
#             if self.verbose:
#                 print(f"  Error in inference: {e}")
#             return None, None


# class EndToEndPipeline:
#     def __init__(self, config_path="config.yaml", verbose=True):
#         self.verbose = verbose
#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)
#         self.dataset_path = self.config.get("dataset_path", "datasets")
#         self.global_cpd_len_path = self.config.get("global_cpd_len_path", "global_datasets/global_cpd_len.txt")
#         self.inference_output_dir = self.config.get("inference_output_dir", "saved_inference_outputs")
#         self.dataset_dir = self.dataset_path
#         self.cpd_len_dir = os.path.dirname(self.global_cpd_len_path) or "."
#         self.bif_dir = "dataset_bif_files"
#         self.json_dir = "bif_json"
#         os.makedirs(self.dataset_dir, exist_ok=True)
#         os.makedirs(self.cpd_len_dir, exist_ok=True)
#         os.makedirs(self.inference_output_dir, exist_ok=True)
#         os.makedirs(self.bif_dir, exist_ok=True)
#         os.makedirs(self.json_dir, exist_ok=True)

#     def run(self, networks=None, train_split=None, val_split=None):
#         if train_split is None:
#             train_split = self.config.get("train_split", 0.7)
#         if val_split is None:
#             val_split = self.config.get("val_split", 0.2)
#         print("=" * 60)
#         print("BIF → GNN DATASET PIPELINE")
#         print("=" * 60)
#         print("\n[1/5] Downloading BIF files...")
#         downloader = BIFDownloader(self.bif_dir, verbose=self.verbose)
#         bif_paths = downloader.download_multiple(networks)
#         print(f"✓ Downloaded {len(bif_paths)} BIF files")
#         print("\n[2/5] Converting BIF to JSON...")
#         converter = BIFToGraphConverter(verbose=self.verbose)
#         json_paths = []
#         for bif_path in bif_paths:
#             name = Path(bif_path).stem
#             model, _ = converter.load_bif(bif_path)
#             json_path = converter.convert_to_json(model, name, self.json_dir)
#             json_paths.append(json_path)
#         print(f"✓ Created {len(json_paths)} JSON files")
#         print("\n[3/5] Creating PyG graphs...")

#         # Determine max CPD length globally
#         all_cpd_lens = []
#         for json_path in json_paths:
#             with open(json_path, 'r') as f:
#                 json_data = json.load(f)
#             for n in json_data["nodes"]:
#                 all_cpd_lens.append(len(json_data["nodes"][n]["cpd"]["values"]))
#         global_cpd_len = max(all_cpd_lens) if all_cpd_lens else 0

#         graphs = []
#         inference_results = []
#         inference_engine = AutoInferenceEngine(
#             max_evidence_nodes=self.config.get("max_evidence_nodes", 3),
#             max_evidence_combinations=self.config.get("max_evidence_combinations", 5),
#             verbose=self.verbose
#         )

#         for idx, json_path in enumerate(json_paths):
#             with open(json_path, 'r') as f:
#                 json_data = json.load(f)
#             model, _ = converter.load_bif(os.path.join(self.bif_dir, f"{json_data['network_name']}.bif"))
#             extractor = GraphFeatureExtractor(verbose=self.verbose)
#             data, _ = extractor.create_graph_dataset(model, json_data, global_cpd_len=global_cpd_len)
            
#             # AUTO INFERENCE - extracts probabilities from actual BIF CPDs
#             prob_dict, evidence = inference_engine.run_inference(model, json_data)
            
#             if prob_dict is not None:
#                 # Get first state probability as y label
#                 first_state = list(prob_dict.keys())[0]
#                 y_value = prob_dict[first_state]
#                 data.y = torch.tensor([y_value], dtype=torch.float)
#             else:
#                 # Fallback
#                 data.y = torch.tensor([0.5], dtype=torch.float)
            
#             graphs.append(data)
            
#             inference_results.append({
#                 "graph_idx": idx,
#                 "graph_name": json_data['network_name'],
#                 "prob": prob_dict if prob_dict else {},
#                 "evidence": evidence if evidence else {},
#                 "mask_strategy": "auto_inference"
#             })
            
#             if self.verbose and (idx < 3 or idx % 5 == 0):
#                 print(f"  [{idx}] {json_data['network_name']}: y={data.y.item():.4f}, evidence={len(evidence) if evidence else 0} nodes")
        
#         print(f"✓ Created {len(graphs)} graphs with auto-inference")
#         print(f"  Global CPD length: {global_cpd_len}")
#         print("\n[4/5] Adding enhanced features...")
#         enhancer = EnhancedGraphPreprocessor(verbose=self.verbose)
#         enhanced_graphs = []
#         for idx, graph in enumerate(graphs):
#             enhanced = enhancer.add_missing_features(graph, global_cpd_len, graph_idx=idx)
#             enhanced_graphs.append(enhanced)
#         print(f"✓ Enhanced {len(enhanced_graphs)} graphs")
#         print("\n[5/5] Splitting and saving dataset...")
#         random.shuffle(enhanced_graphs)
#         n = len(enhanced_graphs)
#         train_end = int(train_split * n)
#         val_end = int((train_split + val_split) * n)
#         train_data = enhanced_graphs[:train_end]
#         val_data = enhanced_graphs[train_end:val_end]
#         test_data = enhanced_graphs[val_end:]
#         train_path = os.path.join(self.dataset_dir, "train.pt")
#         val_path = os.path.join(self.dataset_dir, "val.pt")
#         test_path = os.path.join(self.dataset_dir, "test.pt")
#         torch.save(train_data, train_path)
#         torch.save(val_data, val_path)
#         torch.save(test_data, test_path)
#         with open(self.global_cpd_len_path, 'w') as f:
#             f.write(str(int(global_cpd_len)))
#         metadata = {
#             "total_graphs": len(enhanced_graphs),
#             "train": len(train_data),
#             "val": len(val_data),
#             "test": len(test_data),
#             "global_cpd_len": int(global_cpd_len),
#             "total_features": enhanced_graphs[0].x.shape[1] if enhanced_graphs else 0,
#             "feature_names": [
#                 "node_type", "in_degree", "out_degree", "betweenness", "closeness",
#                 "pagerank", "degree_centrality", "variable_card", "num_parents",
#                 "evidence_flag", f"cpd_0...cpd_{global_cpd_len - 1}",
#                 "cpd_entropy", "evidence_strength", "distance_to_evidence"
#             ]
#         }
#         metadata_path = os.path.join(self.dataset_dir, "metadata.json")
#         with open(metadata_path, 'w') as f:
#             json.dump(metadata, f, indent=2)
#         inference_path = os.path.join(self.inference_output_dir, "inference_results.json")
#         with open(inference_path, "w") as f:
#             json.dump(inference_results, f, indent=2)
#         print(f"✓ Saved datasets:")
#         print(f"  Train: {train_path} ({len(train_data)} graphs)")
#         print(f"  Val: {val_path} ({len(val_data)} graphs)")
#         print(f"  Test: {test_path} ({len(test_data)} graphs)")
#         print(f"  Global CPD Length: {self.global_cpd_len_path}")
#         print(f"  Metadata: {metadata_path}")
#         print(f"  Inference Results saved to: {inference_path}")
#         print("\n" + "=" * 60)
#         print("PIPELINE COMPLETE!")
#         print("=" * 60)
#         return train_data, val_data, test_data, metadata


# if __name__ == "__main__":
#     pipeline = EndToEndPipeline(config_path="config.yaml", verbose=True)
#     bif_files = glob.glob(os.path.join("dataset_bif_files", "*.bif"))
#     networks = [os.path.splitext(os.path.basename(f))[0] for f in bif_files]
#     train, val, test, metadata = pipeline.run(networks=networks)
#     print("\nDataset ready for GNN training!")
#     print(f"Sample graph shape: {train[0].x.shape}")
#     print(f"Sample graph edges: {train[0].edge_index.shape}")
#     print(f"\nDataset Summary:")
#     print(f"  Total graphs: {metadata['total_graphs']}")
#     print(f"  Train: {metadata['train']} graphs")
#     print(f"  Val: {metadata['val']} graphs")
#     print(f"  Test: {metadata['test']} graphs")
#     print(f"  Global CPD Length: {metadata['global_cpd_len']}")
#     print(f"  Total Features per node: {metadata['total_features']}")

"""
Benchmark Script for Pre-trained GraphSAGE Model
Tests a saved model against all BNLearn benchmark networks
"""

import os
import json
import torch
import yaml
import numpy as np
import networkx as nx
import scipy.stats
from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination
from torch_geometric.data import Data
from pathlib import Path
import glob
import torch.nn.functional as F
from graphsage_model import GraphSAGE  # Import your GraphSAGE implementation here
from gat_model import GAT  # Import your GAT implementation here


class BenchmarkDatasetProcessor:
    """
    Process BIF files into graph data using CPD summary features.
    """
    def __init__(self, config_path="config.yaml", verbose=True):
        self.verbose = verbose
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        # CPD summary length fixed to 10
        self.global_cpd_len = 10
        if self.verbose:
            print("Using fixed CPD summary length: 10")

    def load_bif(self, bif_path: str):
        reader = BIFReader(bif_path)
        return reader.get_model()

    def compute_structural_features(self, G: nx.DiGraph, node: str):
        in_deg = G.in_degree[node]
        out_deg = G.out_degree[node]
        try:
            betweenness = nx.betweenness_centrality(G).get(node, 0.0)
            closeness = nx.closeness_centrality(G).get(node, 0.0)
            pagerank = nx.pagerank(G).get(node, 0.0)
            degree_cent = nx.degree_centrality(G).get(node, 0.0)
        except Exception:
            betweenness = closeness = pagerank = degree_cent = 0.0
        return [
            float(in_deg),
            float(out_deg),
            float(betweenness),
            float(closeness),
            float(pagerank),
            float(degree_cent),
        ]

    def get_node_types(self, model):
        roots, intermediates, leaves = [], [], []
        for node in model.nodes():
            parents = list(model.predecessors(node))
            children = list(model.successors(node))
            if len(parents) == 0 and len(children) > 0:
                roots.append(node)
            elif len(parents) > 0 and len(children) > 0:
                intermediates.append(node)
            else:
                leaves.append(node)
        return {"roots": roots, "intermediates": intermediates, "leaves": leaves}

    def extract_cpd_info(self, model, node: str):
        cpd = model.get_cpds(node)
        if cpd is None:
            return {"variable_card": 2, "evidence": [], "values": [0.5, 0.5]}
        return {
            "variable_card": cpd.variable_card,
            "evidence": list(cpd.variables[1:]) if len(cpd.variables) > 1 else [],
            "values": cpd.values.flatten().tolist(),
        }

    def compute_cpd_summary_features(self, cpd_values):
        arr = np.array(cpd_values)
        if arr.sum() == 0:
            arr = np.ones_like(arr) / len(arr)
        else:
            arr = arr / arr.sum()

        return [
            np.mean(arr),
            np.std(arr),
            np.min(arr),
            np.max(arr),
            scipy.stats.entropy(arr),
            float(np.argmax(arr)),
            float(np.count_nonzero(arr)),
            np.median(arr),
            np.percentile(arr, 25),
            np.percentile(arr, 75),
        ]

    def extract_node_features(self, model, G, node, node_type: int):
        cpd_info = self.extract_cpd_info(model, node)
        variable_card = cpd_info["variable_card"]
        num_parents = len(cpd_info["evidence"])

        struct_feat = self.compute_structural_features(G, node)
        cpd_feats = self.compute_cpd_summary_features(cpd_info["values"])

        # Final per-node features:
        # node_type (1) + structural (6) + variable_card (1) + num_parents (1) + evidence_flag (1, default 0)
        # + CPD summary (10)
        features = (
            [float(node_type)]
            + struct_feat
            + [float(variable_card), float(num_parents), 0.0]  # evidence_flag default 0
            + cpd_feats
        )
        return np.array(features, dtype=np.float32)

    def run_inference(self, model, root_node, evidence_nodes=None):
        try:
            inference = VariableElimination(model)
            if not evidence_nodes:
                query_result = inference.query(variables=[root_node])
                probs = query_result.values
                return float(probs[0])
            for node in evidence_nodes[:3]:
                try:
                    states = list(model.get_cpds(node).state_names[node])
                    evidence = {node: states[0]}
                    query_result = inference.query(variables=[root_node], evidence=evidence, show_progress=False)
                    probs = query_result.values
                    return float(probs[0])
                except Exception:
                    continue
            query_result = inference.query(variables=[root_node])
            probs = query_result.values
            return float(probs[0])
        except Exception as e:
            if self.verbose:
                print(f"Warning: Inference failed for {root_node}: {e}")
            return 0.5

    def process_bif_to_graph(self, bif_path: str, network_name: str):
        model = self.load_bif(bif_path)
        node_types = self.get_node_types(model)
        roots = node_types["roots"]
        if len(roots) == 0:
            raise ValueError(f"No root node found in {network_name}")
        root_node = roots[0]

        # Build graph
        G = nx.DiGraph()
        G.add_nodes_from(model.nodes())
        G.add_edges_from(model.edges())

        nodes = sorted(list(model.nodes()))
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        node_features = []
        for node in nodes:
            if node in roots:
                node_type = 0
            elif node in node_types["intermediates"]:
                node_type = 1
            else:
                node_type = 2
            feats = self.extract_node_features(model, G, node, node_type)
            node_features.append(feats)

        x = torch.tensor(np.array(node_features), dtype=torch.float32)

        edge_list = [[node_to_idx[src], node_to_idx[tgt]] for src, tgt in model.edges()]
        edge_index = (torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                      if edge_list else torch.zeros((2, 0), dtype=torch.long))

        # Run inference for ground truth prediction
        non_root_nodes = [n for n in nodes if n not in roots]
        y_true = self.run_inference(model, root_node, non_root_nodes)

        # Append evidence strength and distance to evidence features with defaults
        num_nodes = x.size(0)
        evidence_strength = torch.zeros((num_nodes, 1), dtype=torch.float32)
        distance_to_evidence = torch.full((num_nodes, 1), num_nodes, dtype=torch.float32)

        x_enhanced = torch.cat([x, evidence_strength, distance_to_evidence], dim=1)

        data = Data(x=x_enhanced, edge_index=edge_index, y=torch.tensor([y_true]))

        metadata = {
            "network_name": network_name,
            "num_nodes": len(nodes),
            "num_edges": len(edge_list),
            "root_node": root_node,
            "ground_truth_prob": y_true,
            "num_features": x_enhanced.shape[1],
        }

        if self.verbose:
            print(f"  {network_name}: {len(nodes)} nodes | features={x_enhanced.shape[1]} | GT={y_true:.4f}")

        return data, metadata


class ModelBenchmark:
    """
    Benchmarks a pre-trained GraphSAGE model on test datasets
    """
    def __init__(self, model_path: str, config_path: str = "config.yaml", device=None):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = self.config.get("mode", "root_probability")

        # CPD summary length fixed to 10
        self.global_cpd_len = 10

        # Determine output channels based on mode
        if self.mode == "distribution":
            out_channels = 2
        elif self.mode == "root_probability":
            out_channels = 1
        else:  # regression
            out_channels = self.global_cpd_len

        # in_channels = 1(node_type) + 6(structural) + 1(var_card) + 1(num_parents) + 1(evidence_flag)
        # + 10 (CPD summary) + 2 (evidence strength + distance)
        in_channels = 1 + 6 + 1 + 1 + 1 + 10 + 2  # = 22

        self.model = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=self.config.get("hidden_channels", 32),
            out_channels=out_channels,
            dropout=self.config.get("dropout", 0.5)
        ).to(self.device)

        # self.model = GAT(
        #     in_channels=in_channels,
        #     hidden_channels=self.config.get("hidden_channels", 32),
        #     out_channels=out_channels,
        #     dropout=self.config.get("dropout", 0.2),
        #     heads=8  
        # ).to(device)

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        print(f"✓ Loaded model from {model_path}")
        print(f"  Mode: {self.mode}")
        print(f"  Device: {self.device}")
        print(f"  Input features: {in_channels}")
        print(f"  Output channels: {out_channels}")

    # def predict_single_graph(self, data: Data) -> float:
    #     data = data.to(self.device)
    #     data.batch = torch.zeros(data.x.size(0), dtype=torch.long).to(self.device)
    #     with torch.no_grad():
    #         out = self.model(data)
    #         if self.mode == "root_probability":
    #             pred = torch.sigmoid(out).squeeze().item()
    #         elif self.mode == "distribution":
    #             pred = F.softmax(out, dim=1)[0, 0].item()
    #         else:
    #             pred = out[0].item()
    #     return pred
    def predict_single_graph(self, data: Data) -> float:
        # Move graph and model to same device
        data = data.to(self.device)
        self.model.to(self.device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(data)
            if isinstance(out, tuple):  # GAT returns (x, att_weights)
                out = out[0]

            if self.mode == "root_probability":
                pred = torch.sigmoid(out).squeeze().item()
            elif self.mode == "distribution":
                pred = F.softmax(out, dim=1)[0, 0].item()
            else:
                pred = out.squeeze().item()

        return pred


    def evaluate_dataset(self, graphs, metadata_list):
        predictions, ground_truths, errors, network_results = [], [], [], []

        for graph, meta in zip(graphs, metadata_list):
            pred = self.predict_single_graph(graph)
            true = graph.y.item()

            predictions.append(pred)
            ground_truths.append(true)
            errors.append(abs(pred - true))

            network_results.append({
                "network_name": meta["network_name"],
                "num_nodes": meta["num_nodes"],
                "num_edges": meta["num_edges"],
                "ground_truth": true,
                "prediction": pred,
                "absolute_error": abs(pred - true),
                "relative_error": abs(pred - true) / (true + 1e-8),
            })

        import numpy as np
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        errors = np.array(errors)

        metrics = {
            "mae": np.mean(errors),
            "rmse": np.sqrt(np.mean(errors ** 2)),
            "max_error": np.max(errors),
            "min_error": np.min(errors),
            "median_error": np.median(errors),
            "std_error": np.std(errors),
        }

        if self.mode == "root_probability":
            binary_preds = (predictions > 0.5).astype(int)
            binary_trues = (ground_truths > 0.5).astype(int)
            metrics["accuracy"] = np.mean(binary_preds == binary_trues)

            diff = ground_truths - predictions
            underpredict_mask = diff > 0
            metrics["underpredict_rate"] = np.mean(underpredict_mask)
            metrics["avg_underpredict_error"] = np.mean(diff[underpredict_mask]) if np.any(underpredict_mask) else 0.0

        return {
            "aggregate_metrics": metrics,
            "per_network_results": network_results,
            "predictions": predictions.tolist(),
            "ground_truths": ground_truths.tolist()
        }

    def visualize_results(self, results, output_dir="benchmark_results"):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        os.makedirs(output_dir, exist_ok=True)
        per_network = results["per_network_results"]

        # Scatter plot of true vs predicted
        plt.figure(figsize=(8, 8))
        truths = [r["ground_truth"] for r in per_network]
        preds = [r["prediction"] for r in per_network]
        plt.scatter(truths, preds, alpha=0.6, s=100)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        plt.xlabel('Ground Truth Probability')
        plt.ylabel('Predicted Probability')
        plt.title('Predictions vs Ground Truth')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'scatter_pred_vs_true.png'))
        plt.close()

        # Histogram of errors
        plt.figure(figsize=(10, 6))
        errors = [r["absolute_error"] for r in per_network]
        plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
        plt.close()

        print(f"✓ Saved visualizations to {output_dir}/")


def main():
    print("=" * 70)
    print("GRAPHSAGE MODEL BENCHMARKING ON BNLEARN NETWORKS")
    # print("GAT MODEL BENCHMARKING ON BNLEARN NETWORKS")
    print("=" * 70)

    config_path = "config.yaml"
    model_path = "models/graphsage_root_probability_evidence_only_intermediate.pt"
    # model_path = "models/gatroot_probability_evidence_only_intermediate.pt"
    bif_directory = "dataset_bif_files"
    output_dir = "benchmark_results"

    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("Available models:")
        for model_file in glob.glob("models/*.pt"):
            print(f"  - {model_file}")
        return

    print("\n[1/4] Initializing dataset processor...")
    processor = BenchmarkDatasetProcessor(config_path=config_path, verbose=True)

    bif_files = glob.glob(os.path.join(bif_directory, "*.bif"))
    print(f"\n[2/4] Found {len(bif_files)} BIF files to process")

    print("\n[3/4] Processing BIF files into graphs...")
    graphs = []
    metadata_list = []
    for i, bif_path in enumerate(bif_files, 1):
        network_name = Path(bif_path).stem
        print(f"  [{i}/{len(bif_files)}] Processing {network_name}...", end=" ")
        try:
            graph, meta = processor.process_bif_to_graph(bif_path, network_name)
            graphs.append(graph)
            metadata_list.append(meta)
            print(f"✓ ({meta['num_nodes']} nodes, {meta['num_edges']} edges, GT={meta['ground_truth_prob']:.4f})")
        except Exception as e:
            print(f"✗ Failed: {e}")

    print(f"\n✓ Successfully processed {len(graphs)}/{len(bif_files)} networks")

    print("\n[4/4] Running benchmark...")
    benchmark = ModelBenchmark(model_path=model_path, config_path=config_path)

    results = benchmark.evaluate_dataset(graphs, metadata_list)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    metrics = results["aggregate_metrics"]
    print(f"\nAggregate Metrics:")
    print(f"  MAE:           {metrics['mae']:.4f}")
    print(f"  RMSE:          {metrics['rmse']:.4f}")
    print(f"  Max Error:     {metrics['max_error']:.4f}")
    print(f"  Median Error:  {metrics['median_error']:.4f}")
    print(f"  Std Error:     {metrics['std_error']:.4f}")

    if "accuracy" in metrics:
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:              {metrics['accuracy']:.4f}")
        print(f"  Underpredict Rate:     {metrics['underpredict_rate']:.4f}")
        print(f"  Avg Underpredict Err:  {metrics['avg_underpredict_error']:.4f}")

    # Print best and worst performers omitted for brevity

    results_path = os.path.join(output_dir, "benchmark_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved detailed results to {results_path}")

    benchmark.visualize_results(results, output_dir=output_dir)

    print("\n" + "=" * 70)
    print("BENCHMARKING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
