
# """
# Optimized Benchmark Script for Pre-trained GraphSAGE Model
# Uses caching to speed up graph processing.
# Now includes expanded metrics, tolerance accuracies, error breakdowns, and calibration analysis for root_probability mode.
# """

# import os
# import json
# import torch
# import yaml
# import numpy as np
# import networkx as nx
# import scipy.stats
# from pgmpy.readwrite import BIFReader
# from pgmpy.inference import VariableElimination
# from torch_geometric.data import Data
# from pathlib import Path
# import glob
# import torch.nn.functional as F
# from graphsage_model import GraphSAGE
# from gat_model import GAT
# import hashlib
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score

# class BenchmarkDatasetProcessor:
#     """
#     Process BIF files into graph data using CPD summary features.
#     Now with intelligent caching!
#     """
#     def __init__(self, config_path="config.yaml", verbose=True, cache_dir="cached_graphs"):
#         self.verbose = verbose
#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)
#         # CPD summary length fixed to 10
#         self.global_cpd_len = 10
#         self.cache_dir = cache_dir
#         os.makedirs(cache_dir, exist_ok=True)
#         if self.verbose:
#             print("Using fixed CPD summary length: 10")
#             print(f"Cache directory: {cache_dir}")

#     def _get_cache_path(self, network_name: str) -> str:
#         """Get the cache file path for a network"""
#         return os.path.join(self.cache_dir, f"{network_name}_graph.pt")

#     def _get_metadata_cache_path(self, network_name: str) -> str:
#         """Get the metadata cache file path for a network"""
#         return os.path.join(self.cache_dir, f"{network_name}_metadata.json")

#     def _is_cached(self, network_name: str) -> bool:
#         """Check if both graph and metadata are cached"""
#         graph_path = self._get_cache_path(network_name)
#         meta_path = self._get_metadata_cache_path(network_name)
#         return os.path.exists(graph_path) and os.path.exists(meta_path)

#     def _load_from_cache(self, network_name: str):
#         """Load graph and metadata from cache"""
#         graph_path = self._get_cache_path(network_name)
#         meta_path = self._get_metadata_cache_path(network_name)
        
#         graph = torch.load(graph_path)
#         with open(meta_path, 'r') as f:
#             metadata = json.load(f)
        
#         if self.verbose:
#             print(f"✓ Loaded from cache ({metadata['num_nodes']} nodes, {metadata['num_edges']} edges)")
        
#         return graph, metadata

#     def _save_to_cache(self, network_name: str, graph: Data, metadata: dict):
#         """Save graph and metadata to cache"""
#         graph_path = self._get_cache_path(network_name)
#         meta_path = self._get_metadata_cache_path(network_name)
        
#         torch.save(graph, graph_path)
#         with open(meta_path, 'w') as f:
#             json.dump(metadata, f, indent=2)

#     def load_bif(self, bif_path: str):
#         reader = BIFReader(bif_path)
#         return reader.get_model()

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
#         return [
#             float(in_deg),
#             float(out_deg),
#             float(betweenness),
#             float(closeness),
#             float(pagerank),
#             float(degree_cent),
#         ]

#     def get_node_types(self, model):
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

#     def extract_cpd_info(self, model, node: str):
#         cpd = model.get_cpds(node)
#         if cpd is None:
#             return {"variable_card": 2, "evidence": [], "values": [0.5, 0.5]}
#         return {
#             "variable_card": cpd.variable_card,
#             "evidence": list(cpd.variables[1:]) if len(cpd.variables) > 1 else [],
#             "values": cpd.values.flatten().tolist(),
#         }

#     def compute_cpd_summary_features(self, cpd_values):
#         arr = np.array(cpd_values)
#         if arr.sum() == 0:
#             arr = np.ones_like(arr) / len(arr)
#         else:
#             arr = arr / arr.sum()

#         return [
#             np.mean(arr),
#             np.std(arr),
#             np.min(arr),
#             np.max(arr),
#             scipy.stats.entropy(arr),
#             float(np.argmax(arr)),
#             float(np.count_nonzero(arr)),
#             np.median(arr),
#             np.percentile(arr, 25),
#             np.percentile(arr, 75),
#         ]

#     def extract_node_features(self, model, G, node, node_type: int):
#         cpd_info = self.extract_cpd_info(model, node)
#         variable_card = cpd_info["variable_card"]
#         num_parents = len(cpd_info["evidence"])

#         struct_feat = self.compute_structural_features(G, node)
#         cpd_feats = self.compute_cpd_summary_features(cpd_info["values"])

#         features = (
#             [float(node_type)]
#             + struct_feat
#             + [float(variable_card), float(num_parents), 0.0]
#             + cpd_feats
#         )
#         return np.array(features, dtype=np.float32)

#     def run_inference(self, model, root_node, evidence_nodes=None):
#         try:
#             inference = VariableElimination(model)
#             if not evidence_nodes:
#                 query_result = inference.query(variables=[root_node])
#                 probs = query_result.values
#                 return float(probs[0])
#             for node in evidence_nodes[:3]:
#                 try:
#                     states = list(model.get_cpds(node).state_names[node])
#                     evidence = {node: states[0]}
#                     query_result = inference.query(variables=[root_node], evidence=evidence, show_progress=False)
#                     probs = query_result.values
#                     return float(probs[0])
#                 except Exception:
#                     continue
#             query_result = inference.query(variables=[root_node])
#             probs = query_result.values
#             return float(probs[0])
#         except Exception as e:
#             if self.verbose:
#                 print(f"Warning: Inference failed for {root_node}: {e}")
#             return 0.5

#     def process_bif_to_graph(self, bif_path: str, network_name: str, use_cache=True):
#         """
#         Process BIF to graph with caching support

#         Args:
#             bif_path: Path to BIF file
#             network_name: Name of the network
#             use_cache: Whether to use cached results if available
#         """
#         # Check cache first
#         if use_cache and self._is_cached(network_name):
#             return self._load_from_cache(network_name)

#         # If not cached, process the BIF file
#         if self.verbose:
#             print(f"Processing {network_name}...", end=" ")

#         model = self.load_bif(bif_path)
#         node_types = self.get_node_types(model)
#         roots = node_types["roots"]
#         if len(roots) == 0:
#             raise ValueError(f"No root node found in {network_name}")
#         root_node = roots[0]

#         # Build graph
#         G = nx.DiGraph()
#         G.add_nodes_from(model.nodes())
#         G.add_edges_from(model.edges())

#         nodes = sorted(list(model.nodes()))
#         node_to_idx = {n: i for i, n in enumerate(nodes)}

#         node_features = []
#         for node in nodes:
#             if node in roots:
#                 node_type = 0
#             elif node in node_types["intermediates"]:
#                 node_type = 1
#             else:
#                 node_type = 2
#             feats = self.extract_node_features(model, G, node, node_type)
#             node_features.append(feats)

#         x = torch.tensor(np.array(node_features), dtype=torch.float32)

#         edge_list = [[node_to_idx[src], node_to_idx[tgt]] for src, tgt in model.edges()]
#         edge_index = (torch.tensor(edge_list, dtype=torch.long).t().contiguous()
#                       if edge_list else torch.zeros((2, 0), dtype=torch.long))

#         # Run inference for ground truth prediction
#         non_root_nodes = [n for n in nodes if n not in roots]
#         y_true = self.run_inference(model, root_node, non_root_nodes)

#         # Append evidence strength and distance to evidence features with defaults
#         num_nodes = x.size(0)
#         evidence_strength = torch.zeros((num_nodes, 1), dtype=torch.float32)
#         distance_to_evidence = torch.full((num_nodes, 1), num_nodes, dtype=torch.float32)

#         x_enhanced = torch.cat([x, evidence_strength, distance_to_evidence], dim=1)

#         data = Data(x=x_enhanced, edge_index=edge_index, y=torch.tensor([y_true]))

#         metadata = {
#             "network_name": network_name,
#             "num_nodes": len(nodes),
#             "num_edges": len(edge_list),
#             "root_node": root_node,
#             "ground_truth_prob": y_true,
#             "num_features": x_enhanced.shape[1],
#         }

#         # Save to cache
#         if use_cache:
#             self._save_to_cache(network_name, data, metadata)

#         if self.verbose:
#             print(f"✓ ({len(nodes)} nodes, {len(edge_list)} edges, GT={y_true:.4f})")

#         return data, metadata


# def compute_rootprob_advanced_metrics(preds, trues):
#     """Compute expanded calibration, tolerance, and safety metrics for root_probability mode."""
#     metrics = dict()
#     # Standard regression metrics
#     metrics["mae"] = np.mean(np.abs(preds - trues))
#     metrics["rmse"] = np.sqrt(np.mean((preds - trues)**2))
#     metrics["mse"] = np.mean((preds - trues)**2)
#     metrics["r2_score"] = r2_score(trues, preds)
#     # Tolerance-based accuracy
#     for tol in [0.05, 0.10, 0.15]:
#         within_tolerance = np.abs(preds - trues) <= tol
#         metrics[f"accuracy_within_{int(tol*100)}pct"] = np.mean(within_tolerance)
#     # Asymmetric error analysis
#     errors = trues - preds
#     under_mask = errors > 0
#     over_mask = errors < 0
#     metrics["underpredict_rate"] = np.mean(under_mask)
#     metrics["overpredict_rate"] = np.mean(over_mask)
#     metrics["mean_underpredict_error"] = np.mean(errors[under_mask]) if np.any(under_mask) else 0.0
#     metrics["mean_overpredict_error"] = np.mean(np.abs(errors[over_mask])) if np.any(over_mask) else 0.0
#     # High/low risk analysis
#     high_risk_mask = trues > 0.7
#     metrics["high_risk_mae"] = np.mean(np.abs(preds[high_risk_mask] - trues[high_risk_mask])) if np.any(high_risk_mask) else 0.0
#     metrics["high_risk_underpredict_rate"] = np.mean(preds[high_risk_mask] < trues[high_risk_mask]) if np.any(high_risk_mask) else 0.0
#     low_risk_mask = trues < 0.3
#     metrics["low_risk_mae"] = np.mean(np.abs(preds[low_risk_mask] - trues[low_risk_mask])) if np.any(low_risk_mask) else 0.0
#     # Percentile errors
#     abs_errors = np.abs(preds - trues)
#     metrics["p50_error"] = np.percentile(abs_errors, 50)
#     metrics["p95_error"] = np.percentile(abs_errors, 95)
#     metrics["p99_error"] = np.percentile(abs_errors, 99)
#     # Calibration metrics
#     metrics["mean_prediction"] = np.mean(preds)
#     metrics["mean_ground_truth"] = np.mean(trues)
#     metrics["mean_bias"] = np.mean(preds) - np.mean(trues)
#     # For legacy outputs
#     metrics["max_error"] = np.max(abs_errors)
#     metrics["min_error"] = np.min(abs_errors)
#     metrics["median_error"] = np.median(abs_errors)
#     metrics["std_error"] = np.std(abs_errors)
#     # Binary accuracy for threshold == 0.5
#     metrics["accuracy"] = np.mean(((preds > 0.5) == (trues > 0.5)))
#     return metrics


# class ModelBenchmark:
#     """
#     Benchmarks a pre-trained GraphSAGE model on test datasets.
#     Includes extra root_probability metrics & plots.
#     """
#     def __init__(self, model_path: str, config_path: str = "config.yaml", device=None):
#         self.config_path = config_path
#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)
#         self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.mode = self.config.get("mode", "root_probability")
#         self.global_cpd_len = 10
#         if self.mode == "distribution":
#             out_channels = 2
#         elif self.mode == "root_probability":
#             out_channels = 1
#         else:
#             out_channels = self.global_cpd_len
#         in_channels = 1 + 6 + 1 + 1 + 1 + 10 + 2
#         self.model = GraphSAGE(
#             in_channels=in_channels,
#             hidden_channels=self.config.get("hidden_channels", 32),
#             out_channels=out_channels,
#             dropout=self.config.get("dropout", 0.5)
#         ).to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
#         print(f"✓ Loaded model from {model_path}")
#         print(f"  Mode: {self.mode}")
#         print(f"  Device: {self.device}")
#         print(f"  Input features: {in_channels}")
#         print(f"  Output channels: {out_channels}")

#     def predict_single_graph(self, data: Data) -> float:
#         data = data.to(self.device)
#         self.model.to(self.device)
#         data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
#         with torch.no_grad():
#             out = self.model(data)
#             if isinstance(out, tuple):
#                 out = out[0]
#             if self.mode == "root_probability":
#                 pred = torch.sigmoid(out).squeeze().item()
#             elif self.mode == "distribution":
#                 pred = F.softmax(out, dim=1)[0, 0].item()
#             else:
#                 pred = out.squeeze().item()
#         return pred

#     def evaluate_dataset(self, graphs, metadata_list):
#         predictions, ground_truths, errors, network_results = [], [], [], []
#         for graph, meta in zip(graphs, metadata_list):
#             pred = self.predict_single_graph(graph)
#             true = graph.y.item()
#             predictions.append(pred)
#             ground_truths.append(true)
#             errors.append(abs(pred - true))
#             network_results.append({
#                 "network_name": meta["network_name"],
#                 "num_nodes": meta["num_nodes"],
#                 "num_edges": meta["num_edges"],
#                 "ground_truth": true,
#                 "prediction": pred,
#                 "absolute_error": abs(pred - true),
#                 "relative_error": abs(pred - true) / (true + 1e-8),
#             })
#         predictions = np.array(predictions)
#         ground_truths = np.array(ground_truths)
#         errors = np.array(errors)
#         # Use expanded metrics for root_probability
#         if self.mode == "root_probability":
#             metrics = compute_rootprob_advanced_metrics(predictions, ground_truths)
#         else:
#             metrics = {
#                 "mae": np.mean(np.abs(predictions - ground_truths)),
#                 "rmse": np.sqrt(np.mean((predictions - ground_truths) ** 2)),
#                 "max_error": np.max(errors),
#                 "min_error": np.min(errors),
#                 "median_error": np.median(errors),
#                 "std_error": np.std(errors),
#             }
#         return {
#             "aggregate_metrics": metrics,
#             "per_network_results": network_results,
#             "predictions": predictions.tolist(),
#             "ground_truths": ground_truths.tolist()
#         }

#     def visualize_results(self, results, output_dir="benchmark_results"):
#         os.makedirs(output_dir, exist_ok=True)
#         per_network = results["per_network_results"]
#         # True vs Predicted plot
#         truths = [r["ground_truth"] for r in per_network]
#         preds = [r["prediction"] for r in per_network]
#         plt.figure(figsize=(8, 8))
#         plt.scatter(truths, preds, alpha=0.6, s=100)
#         plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
#         plt.xlabel('Ground Truth Probability')
#         plt.ylabel('Predicted Probability')
#         plt.title('Predictions vs Ground Truth')
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, 'scatter_pred_vs_true.png'))
#         plt.close()
#         # Error distribution histogram
#         errors = [r["absolute_error"] for r in per_network]
#         plt.figure(figsize=(10, 6))
#         plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
#         plt.xlabel('Absolute Error')
#         plt.ylabel('Frequency')
#         plt.title('Prediction Error Distribution')
#         plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
#         plt.close()
#         print(f"✓ Saved visualizations to {output_dir}/")
#         # Percentile error bar chart
#         agg = results["aggregate_metrics"]
#         fig, ax = plt.subplots(figsize=(7,5))
#         bars = ["Median", "95th %ile", "99th %ile"]
#         vals = [agg["p50_error"], agg["p95_error"], agg["p99_error"]]
#         ax.bar(bars, vals, color=["gray", "orange", "red"])
#         ax.set_ylabel("Absolute Error")
#         ax.set_title("Error Percentiles")
#         ax.grid(axis='y', alpha=0.2)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'error_percentiles.png'))
#         plt.close()
#         print(f"✓ Saved percentile error chart to {output_dir}/")

# def main():
#     print("=" * 70)
#     print("OPTIMIZED GRAPHSAGE MODEL BENCHMARKING (WITH CACHING)")
#     print("=" * 70)
#     config_path = "config.yaml"
#     model_path = "training_results/models/graphsage_root_probability_evidence_only_intermediate_fold_9.pt"
#     bif_directory = "dataset_bif_files"
#     output_dir = "benchmark_results"
#     cache_dir = "cached_graphs"
#     if not os.path.exists(model_path):
#         print(f"✗ Model not found: {model_path}")
#         print("Available models:")
#         for model_file in glob.glob("models/*.pt"):
#             print(f"  - {model_file}")
#         return
#     print("\n[1/4] Initializing dataset processor with caching...")
#     processor = BenchmarkDatasetProcessor(
#         config_path=config_path, 
#         verbose=True,
#         cache_dir=cache_dir
#     )
#     bif_files = glob.glob(os.path.join(bif_directory, "*.bif"))
#     print(f"\n[2/4] Found {len(bif_files)} BIF files to process")
#     print("\n[3/4] Processing BIF files into graphs (using cache when available)...")
#     graphs = []
#     metadata_list = []
#     cached_count = 0
#     processed_count = 0
#     for i, bif_path in enumerate(bif_files, 1):
#         network_name = Path(bif_path).stem
#         print(f"  [{i}/{len(bif_files)}] {network_name}...", end=" ")
#         try:
#             was_cached = processor._is_cached(network_name)
#             graph, meta = processor.process_bif_to_graph(bif_path, network_name, use_cache=True)
#             graphs.append(graph)
#             metadata_list.append(meta)
#             if was_cached:
#                 cached_count += 1
#             else:
#                 processed_count += 1
#         except Exception as e:
#             print(f"✗ Failed: {e}")
#     print(f"\n✓ Successfully loaded {len(graphs)}/{len(bif_files)} networks")
#     print(f"  • Loaded from cache: {cached_count}")
#     print(f"  • Newly processed: {processed_count}")
#     print("\n[4/4] Running benchmark...")
#     benchmark = ModelBenchmark(model_path=model_path, config_path=config_path)
#     results = benchmark.evaluate_dataset(graphs, metadata_list)
#     print("\n" + "=" * 70)
#     print("BENCHMARK RESULTS")
#     print("=" * 70)
#     metrics = results["aggregate_metrics"]
#     print(f"\nAggregate Metrics:")
#     print(f"  MAE:           {metrics['mae']:.4f}")
#     print(f"  RMSE:          {metrics['rmse']:.4f}")
#     print(f"  Max Error:     {metrics['max_error']:.4f}")
#     print(f"  Median Error:  {metrics['median_error']:.4f}")
#     print(f"  Std Error:     {metrics['std_error']:.4f}")

#     # ----------- ADVANCED METRIC PRINTING FOR ROOT_PROB --------
#     if "r2_score" in metrics:
#         print(f"  R² Score:      {metrics['r2_score']:.4f}")
#         print(f"\n--- Tolerance-Based Accuracy ---")
#         print(f"  Within 5%:     {metrics['accuracy_within_5pct']:.4f}")
#         print(f"  Within 10%:    {metrics['accuracy_within_10pct']:.4f}")
#         print(f"  Within 15%:    {metrics['accuracy_within_15pct']:.4f}")
#         print(f"\n--- Safety-Critical ---")
#         print(f"  Underpredict Rate:        {metrics['underpredict_rate']:.4f}")
#         print(f"  Mean Underpredict Error:  {metrics['mean_underpredict_error']:.4f}")
#         print(f"  High-Risk MAE:            {metrics['high_risk_mae']:.4f}")
#         print(f"  High-Risk Underpredict:   {metrics['high_risk_underpredict_rate']:.4f}")
#         print(f"  Low-Risk MAE:             {metrics['low_risk_mae']:.4f}")
#         print(f"\n--- Error Percentiles ---")
#         print(f"  Median Error:   {metrics['p50_error']:.4f}")
#         print(f"  95%ile Error:   {metrics['p95_error']:.4f}")
#         print(f"  99%ile Error:   {metrics['p99_error']:.4f}")
#         print(f"\n--- Calibration ---")
#         print(f"  Mean Prediction:   {metrics['mean_prediction']:.4f}")
#         print(f"  Mean Ground Truth: {metrics['mean_ground_truth']:.4f}")
#         print(f"  Mean Bias:         {metrics['mean_bias']:.4f}")
#         print(f"\n--- Classification ---")
#         print(f"  Accuracy >0.5:     {metrics['accuracy']:.4f}")

#     results_path = os.path.join(output_dir, "benchmark_results.json")
#     os.makedirs(output_dir, exist_ok=True)
#     with open(results_path, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\n✓ Saved detailed results to {results_path}")
    
#     # Save key metrics to txt for convenience
#     metrics_txt_path = os.path.join(output_dir, "benchmark_metrics.txt")
#     with open(metrics_txt_path, "w") as f:
#         for k, v in metrics.items():
#             f.write(f"{k}: {v:.6f}\n")
#     print(f"✓ Saved metrics to {metrics_txt_path}")

#     benchmark.visualize_results(results, output_dir=output_dir)

#     print("\n" + "=" * 70)
#     print("BENCHMARKING COMPLETE!")
#     print("=" * 70)

# if __name__ == "__main__":
#     main()
"""
Benchmark Script
- Supports 25 features (with graph-level features)
- Size-normalized degrees (matches training)
- Proper log-prob and normal-prob handling
- Cache invalidation on settings change
- Optional network size filtering
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
from graphsage_model import GraphSAGE
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import hashlib


# ===== OPTIONAL: Set to filter large networks =====
MAX_NODES_FILTER = None  # Set to 500 to only test networks <500 nodes, or None to disable


class BenchmarkDatasetProcessor:
    """
    Process BIF files with COMPLETE feature extraction (25 features)
    """
    def __init__(self, config_path="config.yaml", verbose=True, cache_dir="cached_graphs"):
        self.verbose = verbose
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.global_cpd_len = 10
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.use_log_prob = self.config.get("use_log_prob", False)
        
        # Load normalization stats
        self.norm_stats = None
        self._load_normalization_stats()
        
        # Create cache key based on settings
        self.cache_key = self._get_cache_key()
        
        if self.verbose:
            print("Using fixed CPD summary length: 10")
            print(f"Cache directory: {cache_dir}")
            print(f"Log-probability mode: {self.use_log_prob}")
            print(f"Normalization: {'Enabled' if self.norm_stats else 'Disabled'}")
            print(f"Size filter: {MAX_NODES_FILTER if MAX_NODES_FILTER else 'Disabled'}")
            print(f"Cache key: {self.cache_key[:8]}...")

    def _get_cache_key(self):
        """Generate cache key based on settings"""
        key_string = f"logprob_{self.use_log_prob}_norm_{self.norm_stats is not None}_filter_{MAX_NODES_FILTER}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_normalization_stats(self):
        """Load normalization statistics"""
        norm_paths = [
            "datasets/folds/fold_0_norm_stats.pt",
            "datasets/norm_stats.pt"
        ]
        
        for path in norm_paths:
            if os.path.exists(path):
                try:
                    self.norm_stats = torch.load(path, weights_only=False)
                    if self.verbose:
                        print(f"✓ Loaded normalization stats from {path}")
                    return
                except Exception as e:
                    if self.verbose:
                        print(f"⚠ Failed to load {path}: {e}")
        
        if self.verbose:
            print("⚠ WARNING: No normalization stats found!")

    def _normalize_features(self, x, cpd_start_idx=10):
        """Apply same normalization as training"""
        if self.norm_stats is None:
            return x
        
        x = x.clone()
        
        try:
            # Normalize base features (indices 1-8)
            indices = self.norm_stats['base']['indices']
            mean_tensor = torch.tensor(self.norm_stats['base']['mean'], dtype=torch.float)
            std_tensor = torch.tensor(self.norm_stats['base']['std'], dtype=torch.float)
            x[:, indices] = (x[:, indices] - mean_tensor) / std_tensor
            
            # Normalize CPD (skip argmax at index 5 relative to cpd_start)
            cpd = x[:, cpd_start_idx:cpd_start_idx+10]
            cpd_to_norm = torch.cat([cpd[:, :5], cpd[:, 6:]], dim=1)
            cpd_mean = torch.tensor(self.norm_stats['cpd']['mean'], dtype=torch.float)
            cpd_std = torch.tensor(self.norm_stats['cpd']['std'], dtype=torch.float)
            cpd_normed = (cpd_to_norm - cpd_mean) / cpd_std
            x[:, cpd_start_idx:cpd_start_idx+5] = cpd_normed[:, :5]
            x[:, cpd_start_idx+6:cpd_start_idx+10] = cpd_normed[:, 5:]
            
            # Normalize distance (last feature)
            x[:, -1] = (x[:, -1] - self.norm_stats['distance']['mean']) / self.norm_stats['distance']['std']
            
        except Exception as e:
            if self.verbose:
                print(f"⚠ Normalization failed: {e}")
        
        return x

    def _get_cache_path(self, network_name: str) -> str:
        return os.path.join(self.cache_dir, f"{network_name}_{self.cache_key}_graph.pt")

    def _get_metadata_cache_path(self, network_name: str) -> str:
        return os.path.join(self.cache_dir, f"{network_name}_{self.cache_key}_metadata.json")

    def _is_cached(self, network_name: str) -> bool:
        graph_path = self._get_cache_path(network_name)
        meta_path = self._get_metadata_cache_path(network_name)
        return os.path.exists(graph_path) and os.path.exists(meta_path)

    def _load_from_cache(self, network_name: str):
        graph_path = self._get_cache_path(network_name)
        meta_path = self._get_metadata_cache_path(network_name)
        
        graph = torch.load(graph_path, weights_only=False)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        if self.verbose:
            gt_str = f"log(p)={metadata['ground_truth_prob']:.4f}" if self.use_log_prob else f"p={metadata['ground_truth_prob']:.4f}"
            print(f"✓ Cache ({metadata['num_nodes']}N, {metadata['num_edges']}E, GT:{gt_str})")
        
        return graph, metadata

    def _save_to_cache(self, network_name: str, graph: Data, metadata: dict):
        graph_path = self._get_cache_path(network_name)
        meta_path = self._get_metadata_cache_path(network_name)
        
        torch.save(graph, graph_path)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

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
            float(in_deg), float(out_deg), float(betweenness),
            float(closeness), float(pagerank), float(degree_cent),
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
        """Compute 10 CPD summary statistics"""
        arr = np.array(cpd_values)
        if arr.sum() == 0:
            arr = np.ones_like(arr) / len(arr)
        else:
            arr = arr / arr.sum()

        return [
            np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
            scipy.stats.entropy(arr), float(np.argmax(arr)),
            float(np.count_nonzero(arr)), np.median(arr),
            np.percentile(arr, 25), np.percentile(arr, 75),
        ]

    def extract_node_features(self, model, G, node, node_type: int, num_nodes: int):
        """
        Extract node features with SIZE NORMALIZATION
        Returns 20 features (before graph features added):
        [0] node_type
        [1-2] in_deg, out_deg (SIZE-NORMALIZED)
        [3-6] betweenness, closeness, pagerank, degree_cent
        [7] variable_card
        [8] num_parents
        [9] evidence_flag (placeholder)
        [10-19] CPD summary (10 features)
        """
        cpd_info = self.extract_cpd_info(model, node)
        variable_card = cpd_info["variable_card"]
        num_parents = len(cpd_info["evidence"])

        struct_feat = self.compute_structural_features(G, node)
        
        # ===== SIZE-NORMALIZE DEGREES (CRITICAL: MATCH TRAINING!) =====
        struct_feat[0] = struct_feat[0] / num_nodes  # in_degree / num_nodes
        struct_feat[1] = struct_feat[1] / num_nodes  # out_degree / num_nodes
        
        cpd_feats = self.compute_cpd_summary_features(cpd_info["values"])

        features = (
            [float(node_type)] +           # [0]
            struct_feat +                  # [1-6]
            [float(variable_card),         # [7]
             float(num_parents),           # [8]
             0.0] +                        # [9] evidence_flag placeholder
            cpd_feats                      # [10-19]
        )
        return np.array(features, dtype=np.float32)

    def run_inference(self, model, root_node, evidence_nodes=None):
        """Run inference and return in appropriate space (log or prob)"""
        try:
            inference = VariableElimination(model)
            if not evidence_nodes:
                query_result = inference.query(variables=[root_node])
                probs = query_result.values
                prob_value = float(probs[0])
            else:
                for node in evidence_nodes[:3]:
                    try:
                        states = list(model.get_cpds(node).state_names[node])
                        evidence = {node: states[0]}
                        query_result = inference.query(
                            variables=[root_node], 
                            evidence=evidence, 
                            show_progress=False
                        )
                        probs = query_result.values
                        prob_value = float(probs[0])
                        break
                    except Exception:
                        continue
                else:
                    query_result = inference.query(variables=[root_node])
                    probs = query_result.values
                    prob_value = float(probs[0])
            
            # Convert to log-space if needed
            if self.use_log_prob:
                prob_value = max(prob_value, 1e-10)
                return np.log(prob_value)
            else:
                return prob_value
                
        except Exception as e:
            if self.verbose:
                print(f"⚠ Inference failed for {root_node}: {e}")
            return np.log(0.5) if self.use_log_prob else 0.5

    def process_bif_to_graph(self, bif_path: str, network_name: str, use_cache=True):
        """
        Process BIF to graph with ALL 25 features:
        [0-9]: Basic node features
        [10-19]: CPD summary
        [20-22]: Graph features (log_size, density, max_path)
        [23]: evidence_strength
        [24]: distance_to_evidence
        """
        
        # Check cache
        if use_cache and self._is_cached(network_name):
            return self._load_from_cache(network_name)

        # Process from scratch
        if self.verbose:
            print(f"Processing {network_name}...", end=" ")

        model = self.load_bif(bif_path)
        
        # ===== SIZE FILTER =====
        num_nodes = len(model.nodes())
        if MAX_NODES_FILTER is not None and num_nodes > MAX_NODES_FILTER:
            if self.verbose:
                print(f"⊘ Skipped (>{MAX_NODES_FILTER} nodes)")
            return None, None
        
        node_types = self.get_node_types(model)
        roots = node_types["roots"]
        if len(roots) == 0:
            raise ValueError(f"No root node in {network_name}")
        root_node = roots[0]

        # Build graph
        G = nx.DiGraph()
        G.add_nodes_from(model.nodes())
        G.add_edges_from(model.edges())

        nodes = sorted(list(model.nodes()))
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Extract node features (20 features)
        node_features = []
        for node in nodes:
            if node in roots:
                node_type = 0
            elif node in node_types["intermediates"]:
                node_type = 1
            else:
                node_type = 2
            feats = self.extract_node_features(model, G, node, node_type, num_nodes)
            node_features.append(feats)

        x = torch.tensor(np.array(node_features), dtype=torch.float32)

        edge_list = [[node_to_idx[src], node_to_idx[tgt]] for src, tgt in model.edges()]
        edge_index = (torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                      if edge_list else torch.zeros((2, 0), dtype=torch.long))

        # ===== EXTRACT METADATA FOR GRAPH FEATURES =====
        try:
            json_path = bif_path.replace('.bif', '.json')
            if os.path.exists(json_path):
                with open(json_path) as f:
                    json_data = json.load(f)
                metadata = json_data.get("metadata", {})
                num_edges_meta = metadata.get("total_edges", len(edge_list))
                max_path = json_data.get("paths_info", {}).get("max_path_length", 0)
            else:
                num_edges_meta = len(edge_list)
                max_path = 0
        except:
            num_edges_meta = len(edge_list)
            max_path = 0

        # Calculate graph features
        graph_density = num_edges_meta / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0

        # Create graph feature tensor [20-22] (same for all nodes)
        graph_feats = torch.tensor([
            np.log(num_nodes + 1),    # [20] log(graph_size)
            graph_density,             # [21] edge density
            max_path / 10.0            # [22] normalized max path
        ], dtype=torch.float).unsqueeze(0).expand(num_nodes, -1)

        # Run inference
        non_root_nodes = [n for n in nodes if n not in roots]
        y_true = self.run_inference(model, root_node, non_root_nodes)

        # Add auxiliary features [23-24]
        evidence_strength = torch.zeros((num_nodes, 1), dtype=torch.float32)  # [23]
        distance_to_evidence = torch.full((num_nodes, 1), num_nodes, dtype=torch.float32)  # [24]
        
        # Concatenate: [0-19] + [20-22] + [23] + [24] = 25 features
        x_enhanced = torch.cat([x, graph_feats, evidence_strength, distance_to_evidence], dim=1)

        # ===== CRITICAL: Apply normalization =====
        x_enhanced = self._normalize_features(x_enhanced, cpd_start_idx=10)

        # Create Data object
        data = Data(x=x_enhanced, edge_index=edge_index, y=torch.tensor([y_true], dtype=torch.float))

        metadata = {
            "network_name": network_name,
            "num_nodes": num_nodes,
            "num_edges": len(edge_list),
            "root_node": root_node,
            "ground_truth_prob": y_true,
            "num_features": x_enhanced.shape[1],
            "use_log_prob": self.use_log_prob,
            "normalized": self.norm_stats is not None
        }

        # Save to cache
        if use_cache:
            self._save_to_cache(network_name, data, metadata)

        if self.verbose:
            prob_str = f"log(p)={y_true:.4f}" if self.use_log_prob else f"p={y_true:.4f}"
            print(f"✓ ({num_nodes}N, {len(edge_list)}E, GT:{prob_str})")

        return data, metadata


def compute_rootprob_advanced_metrics(preds, trues, use_log_prob=False):
    """Compute comprehensive metrics with log-prob support"""
    metrics = dict()
    
    # Convert to probability space
    if use_log_prob:
        preds_clamped = np.clip(preds, -10, 0)
        trues_clamped = np.clip(trues, -10, 0)
        preds_prob = np.exp(preds_clamped)
        trues_prob = np.exp(trues_clamped)
        
        # Log-space metrics
        metrics["mae_logspace"] = np.mean(np.abs(preds - trues))
        metrics["rmse_logspace"] = np.sqrt(np.mean((preds - trues)**2))
    else:
        preds_prob = preds
        trues_prob = trues
    
    # Probability-space metrics
    metrics["mae"] = np.mean(np.abs(preds_prob - trues_prob))
    metrics["rmse"] = np.sqrt(np.mean((preds_prob - trues_prob)**2))
    metrics["mse"] = np.mean((preds_prob - trues_prob)**2)
    metrics["r2_score"] = r2_score(trues_prob, preds_prob)
    
    # Tolerance accuracy
    for tol in [0.05, 0.10, 0.15]:
        within = np.abs(preds_prob - trues_prob) <= tol
        metrics[f"accuracy_within_{int(tol*100)}pct"] = np.mean(within)
    
    # Asymmetric errors
    errors = trues_prob - preds_prob
    under_mask = errors > 0
    over_mask = errors < 0
    metrics["underpredict_rate"] = np.mean(under_mask)
    metrics["overpredict_rate"] = np.mean(over_mask)
    metrics["mean_underpredict_error"] = np.mean(errors[under_mask]) if np.any(under_mask) else 0.0
    metrics["mean_overpredict_error"] = np.mean(np.abs(errors[over_mask])) if np.any(over_mask) else 0.0
    
    # Risk analysis
    high_risk = trues_prob > 0.7
    metrics["high_risk_mae"] = np.mean(np.abs(preds_prob[high_risk] - trues_prob[high_risk])) if np.any(high_risk) else 0.0
    metrics["high_risk_underpredict_rate"] = np.mean(preds_prob[high_risk] < trues_prob[high_risk]) if np.any(high_risk) else 0.0
    
    low_risk = trues_prob < 0.3
    metrics["low_risk_mae"] = np.mean(np.abs(preds_prob[low_risk] - trues_prob[low_risk])) if np.any(low_risk) else 0.0
    
    # Percentiles
    abs_errors = np.abs(preds_prob - trues_prob)
    metrics["p50_error"] = np.percentile(abs_errors, 50)
    metrics["p95_error"] = np.percentile(abs_errors, 95)
    metrics["p99_error"] = np.percentile(abs_errors, 99)
    
    # Calibration
    metrics["mean_prediction"] = np.mean(preds_prob)
    metrics["mean_ground_truth"] = np.mean(trues_prob)
    metrics["mean_bias"] = np.mean(preds_prob) - np.mean(trues_prob)
    
    # Legacy
    metrics["max_error"] = np.max(abs_errors)
    metrics["min_error"] = np.min(abs_errors)
    metrics["median_error"] = np.median(abs_errors)
    metrics["std_error"] = np.std(abs_errors)
    metrics["accuracy"] = np.mean(((preds_prob > 0.5) == (trues_prob > 0.5)))
    
    return metrics


class ModelBenchmark:
    """Benchmark with proper 25-feature handling"""
    def __init__(self, model_path: str, config_path: str = "config.yaml", device=None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = self.config.get("mode", "root_probability")
        self.use_log_prob = self.config.get("use_log_prob", False)
        self.global_cpd_len = 10

        if self.mode == "distribution":
            out_channels = 2
        elif self.mode == "root_probability":
            out_channels = 1
        else:
            out_channels = self.global_cpd_len

        # ===== CRITICAL: 25 input features =====
        in_channels = 25  # 10 base + 10 CPD + 3 graph + 1 evidence + 1 distance

        self.model = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=self.config.get("hidden_channels", 128),
            out_channels=out_channels,
            dropout=self.config.get("dropout", 0.1),
            mode=self.mode,
            use_log_prob=self.use_log_prob
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.eval()

        print(f"✓ Loaded model from {model_path}")
        print(f"  Mode: {self.mode}, Log-prob: {self.use_log_prob}, Device: {self.device}")
        print(f"  Input features: {in_channels}")

    def predict_single_graph(self, data: Data) -> float:
        data = data.to(self.device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(data)
            if isinstance(out, tuple):
                out = out[0]

            if self.mode == "root_probability":
                pred = out.squeeze().item()  # Already in correct space (log or prob)
            elif self.mode == "distribution":
                pred = F.softmax(out, dim=1)[0, 0].item()
            else:
                pred = out.squeeze().item()

        return pred

    def evaluate_dataset(self, graphs, metadata_list):
        predictions, ground_truths = [], []

        for graph, meta in zip(graphs, metadata_list):
            pred = self.predict_single_graph(graph)
            true = graph.y.item()
            predictions.append(pred)
            ground_truths.append(true)

        preds = np.array(predictions)
        trues = np.array(ground_truths)

        if self.mode == "root_probability":
            metrics = compute_rootprob_advanced_metrics(preds, trues, self.use_log_prob)
        else:
            if self.use_log_prob:
                preds = np.exp(np.clip(preds, -10, 0))
                trues = np.exp(np.clip(trues, -10, 0))
            metrics = {
                "mae": np.mean(np.abs(preds - trues)),
                "rmse": np.sqrt(np.mean((preds - trues) ** 2)),
            }

        # Per-network results
        network_results = []
        for i, (graph, meta) in enumerate(zip(graphs, metadata_list)):
            pred_raw = predictions[i]
            true_raw = ground_truths[i]
            
            # Convert to probability space for interpretable metrics
            if self.use_log_prob:
                pred_prob = np.exp(np.clip(pred_raw, -10, 0))
                true_prob = np.exp(np.clip(true_raw, -10, 0))
            else:
                pred_prob = pred_raw
                true_prob = true_raw
                
            network_results.append({
                "network_name": meta["network_name"],
                "num_nodes": meta["num_nodes"],
                "num_edges": meta["num_edges"],
                "ground_truth_raw": true_raw,
                "prediction_raw": pred_raw,
                "ground_truth_prob": true_prob,
                "prediction_prob": pred_prob,
                "absolute_error": abs(pred_prob - true_prob),
                "use_log_prob": self.use_log_prob
            })

        return {
            "aggregate_metrics": metrics,
            "per_network_results": network_results,
            "predictions": preds.tolist(),
            "ground_truths": trues.tolist()
        }

    def visualize_results(self, results, output_dir="benchmark_results"):
        os.makedirs(output_dir, exist_ok=True)
        per_network = results["per_network_results"]

        truths = np.array([r["ground_truth_prob"] for r in per_network])
        preds = np.array([r["prediction_prob"] for r in per_network])

        truths = np.clip(truths, 0, 1)
        preds = np.clip(preds, 0, 1)

        plt.figure(figsize=(8, 8))
        plt.scatter(truths, preds, alpha=0.6, s=100)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
        plt.xlabel('True Probability', fontsize=12)
        plt.ylabel('Predicted Probability', fontsize=12)
        title = 'True vs Predicted Probabilities'
        if self.use_log_prob:
            title += ' (converted from log-space)'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scatter_pred_vs_true.png'), dpi=300)
        plt.close()

        errors = [r["absolute_error"] for r in per_network]
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Absolute Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Error Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
        plt.close()

        print(f"✓ Saved visualizations to {output_dir}/")


def main():
    print("=" * 70)
    print("FULLY FIXED BENCHMARK (25 features + size normalization)")
    print("=" * 70)

    config_path = "config.yaml"
    model_path = "training_results/models/graphsage_root_probability_evidence_only_intermediate_logprob_fold_4.pt"
    bif_directory = "dataset_bif_files"
    output_dir = "benchmark_results"
    cache_dir = "cached_graphs"

    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return

    print("\n[1/4] Initializing processor...")
    processor = BenchmarkDatasetProcessor(config_path, verbose=True, cache_dir=cache_dir)
    
    if processor.norm_stats is None:
        print("\n" + "!"*70)
        print("WARNING: Normalization stats not loaded!")
        print("!"*70)

    bif_files = glob.glob(os.path.join(bif_directory, "*.bif"))
    print(f"\n[2/4] Found {len(bif_files)} BIF files")

    print("\n[3/4] Processing graphs...")
    graphs, metadata_list = [], []

    for i, bif_path in enumerate(bif_files, 1):
        network_name = Path(bif_path).stem
        print(f"  [{i}/{len(bif_files)}] {network_name}...", end=" ")
        try:
            graph, meta = processor.process_bif_to_graph(bif_path, network_name, use_cache=True)
            if graph is not None:  # Skip filtered networks
                graphs.append(graph)
                metadata_list.append(meta)
        except Exception as e:
            print(f"✗ {e}")

    print(f"\n✓ Loaded {len(graphs)}/{len(bif_files)} networks")
    
    # Debug: Check ground truth distribution
    if graphs:
        gts = [g.y.item() for g in graphs[:5]]
        print(f"\nDEBUG - First 5 ground truths:")
        if processor.use_log_prob:
            print(f"  Log-space: {[f'{g:.4f}' for g in gts]}")
            print(f"  Prob-space: {[f'{np.exp(g):.4f}' for g in gts]}")
        else:
            print(f"  Prob-space: {[f'{g:.4f}' for g in gts]}")

    print("\n[4/4] Running benchmark...")
    benchmark = ModelBenchmark(model_path, config_path)
    results = benchmark.evaluate_dataset(graphs, metadata_list)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    metrics = results["aggregate_metrics"]
    print(f"\nCore Metrics (Probability Space):")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics.get('r2_score', 0):.4f}")
    
    if 'mae_logspace' in metrics:
        print(f"\nLog-Space Metrics (for reference):")
        print(f"  MAE:  {metrics['mae_logspace']:.4f}")
        print(f"  RMSE: {metrics['rmse_logspace']:.4f}")

    if "accuracy_within_10pct" in metrics:
        print(f"\nTolerance Accuracy:")
        print(f"  Within  5%: {metrics.get('accuracy_within_5pct', 0):.4f}")
        print(f"  Within 10%: {metrics['accuracy_within_10pct']:.4f}")
        print(f"  Within 15%: {metrics.get('accuracy_within_15pct', 0):.4f}")
    
    if "underpredict_rate" in metrics:
        print(f"\nSafety Metrics:")
        print(f"  Underpredict Rate: {metrics['underpredict_rate']:.4f}")
        print(f"  High-Risk MAE:     {metrics.get('high_risk_mae', 0):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    benchmark.visualize_results(results, output_dir)
    
    print(f"\n✓ Saved to {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()