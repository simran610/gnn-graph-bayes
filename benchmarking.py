"""
Benchmark Matching EXACT Training Configuration
- Evidence: 2 nodes (1 leaf + 1 intermediate)
- Evidence state: Always first state (0)
- Query: Root node first state probability
"""

import os
import json
import time
import torch
import numpy as np
import networkx as nx
import scipy.stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.sampling import BayesianModelSampling
from torch_geometric.data import Data
import yaml

warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Fix memory leak


class InferenceMethod:
    def __init__(self, name: str):
        self.name = name
    
    def infer(self, model, query_node: str, evidence: dict) -> Tuple[float, float]:
        raise NotImplementedError


class PgmpyVariableElimination(InferenceMethod):
    def __init__(self):
        super().__init__("pgmpy-VE")
    
    def infer(self, model, query_node: str, evidence: dict) -> Tuple[float, float]:
        start = time.time()
        try:
            inference = VariableElimination(model)
            result = inference.query(
                variables=[query_node],
                evidence=evidence if evidence else None,
                show_progress=False
            )
            prob = float(result.values[0])  # First state probability
            elapsed = time.time() - start
            return prob, elapsed
        except Exception:
            return 0.5, time.time() - start


class PgmpyBeliefPropagation(InferenceMethod):
    def __init__(self):
        super().__init__("pgmpy-BP")
    
    def infer(self, model, query_node: str, evidence: dict) -> Tuple[float, float]:
        start = time.time()
        if model.number_of_nodes() > 500:
            return 0.5, 0.0
        try:
            inference = BeliefPropagation(model)
            result = inference.query(
                variables=[query_node],
                evidence=evidence if evidence else None,
                show_progress=False
            )
            prob = float(result.values[0])
            return prob, time.time() - start
        except Exception:
            return 0.5, time.time() - start


class PgmpySampling(InferenceMethod):
    def __init__(self, n_samples: int = 10000):
        super().__init__("pgmpy-Sampling")
        self.n_samples = n_samples
    
    def infer(self, model, query_node: str, evidence: dict) -> Tuple[float, float]:
        start = time.time()
        try:
            sampler = BayesianModelSampling(model)
            if evidence:
                samples = sampler.likelihood_weighted_sample(
                    evidence=list(evidence.items()),
                    size=self.n_samples,
                    show_progress=False
                )
            else:
                samples = sampler.forward_sample(
                    size=self.n_samples,
                    show_progress=False
                )
            
            cpd = model.get_cpds(query_node)
            if hasattr(cpd, 'state_names') and query_node in cpd.state_names:
                first_state = cpd.state_names[query_node][0]
            else:
                first_state = list(samples[query_node].unique())[0]
            
            prob = (samples[query_node] == first_state).mean()
            return float(prob), time.time() - start
        except Exception:
            return 0.5, time.time() - start


class BenchmarkDatasetProcessor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.global_cpd_len = 10
    
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
        return [float(in_deg), float(out_deg), float(betweenness), 
                float(closeness), float(pagerank), float(degree_cent)]
    
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
            np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
            scipy.stats.entropy(arr), float(np.argmax(arr)),
            float(np.count_nonzero(arr)), np.median(arr),
            np.percentile(arr, 25), np.percentile(arr, 75),
        ]
    
    def extract_node_features(self, model, G, node, node_type: int):
        cpd_info = self.extract_cpd_info(model, node)
        variable_card = cpd_info["variable_card"]
        num_parents = len(cpd_info["evidence"])
        struct_feat = self.compute_structural_features(G, node)
        cpd_feats = self.compute_cpd_summary_features(cpd_info["values"])
        
        features = (
            [float(node_type)] + struct_feat +
            [float(variable_card), float(num_parents), 0.0] + cpd_feats
        )
        return np.array(features, dtype=np.float32)


class GNNInference(InferenceMethod):
    """GNN matching exact training configuration"""
    def __init__(self, model_path: str, processor, device='cpu'):
        super().__init__("GNN")
        self.processor = processor
        self.device = device
        
        from graphsage_model import GraphSAGE
        self.model = GraphSAGE(
            in_channels=22,
            hidden_channels=128,
            out_channels=1,
            dropout=0.1
        ).to(device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.eval()
        print("✓ GNN loaded (expects: 2 evidence nodes with state=0)")
    
    def compute_distance_to_evidence(self, G_undirected: nx.Graph, 
                                     evidence_indices: List[int], 
                                     num_nodes: int) -> torch.Tensor:
        if len(evidence_indices) == 0:
            return torch.full((num_nodes, 1), float(num_nodes), dtype=torch.float32)
        
        distances = []
        evidence_set = set(evidence_indices)
        
        for node_idx in range(num_nodes):
            if node_idx in evidence_set:
                distances.append(0.0)
            else:
                min_dist = float(num_nodes)
                for ev_idx in evidence_indices:
                    try:
                        if G_undirected.has_node(node_idx) and G_undirected.has_node(ev_idx):
                            if nx.has_path(G_undirected, node_idx, ev_idx):
                                dist = nx.shortest_path_length(G_undirected, node_idx, ev_idx)
                                min_dist = min(min_dist, dist)
                    except nx.NetworkXError:
                        continue
                distances.append(float(min_dist))
        
        return torch.tensor(distances, dtype=torch.float32).unsqueeze(1)
    
    def infer(self, model, query_node: str, evidence: dict) -> Tuple[float, float]:
        start = time.time()
        try:
            G = nx.DiGraph()
            G.add_nodes_from(model.nodes())
            G.add_edges_from(model.edges())
            G_undirected = G.to_undirected()
            
            nodes = sorted(list(model.nodes()))
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            
            node_types = self.processor.get_node_types(model)
            roots = node_types["roots"]
            intermediates = node_types["intermediates"]
            
            node_features = []
            for node in nodes:
                if node in roots:
                    node_type = 0
                elif node in intermediates:
                    node_type = 1
                else:
                    node_type = 2
                
                feats = self.processor.extract_node_features(model, G, node, node_type)
                
                # Mark evidence nodes (state doesn't matter, just flag)
                if node in evidence:
                    feats[9] = 1.0  # evidence flag
                
                node_features.append(feats)
            
            x = torch.tensor(np.array(node_features), dtype=torch.float32)
            
            edge_list = [[node_to_idx[src], node_to_idx[tgt]] for src, tgt in model.edges()]
            edge_index = (torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                         if edge_list else torch.zeros((2, 0), dtype=torch.long))
            
            num_nodes = x.size(0)
            evidence_indices = [node_to_idx[n] for n in evidence.keys() if n in node_to_idx]
            
            evidence_strength = torch.zeros((num_nodes, 1), dtype=torch.float32)
            for idx in evidence_indices:
                evidence_strength[idx] = 1.0
            
            distance_to_evidence = self.compute_distance_to_evidence(
                G_undirected, evidence_indices, num_nodes
            )
            
            x_enhanced = torch.cat([x, evidence_strength, distance_to_evidence], dim=1)
            
            graph = Data(x=x_enhanced, edge_index=edge_index)
            graph = graph.to(self.device)
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                out = self.model(graph)
                prob = torch.sigmoid(out).squeeze().item()
            
            elapsed = time.time() - start
            return prob, elapsed
            
        except Exception as e:
            print(f"  ⚠ GNN failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.5, time.time() - start


class BenchmarkSuite:
    def __init__(self, bif_directory: str, output_dir: str = "benchmark_results"):
        self.bif_directory = bif_directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.methods: List[InferenceMethod] = [
            PgmpyVariableElimination(),
            PgmpyBeliefPropagation(),
            PgmpySampling(n_samples=10000),
        ]
        
        self.bif_files = list(Path(bif_directory).glob("*.bif"))
        print(f"Found {len(self.bif_files)} BIF files")
    
    def add_gnn_method(self, model_path: str, processor):
        gnn = GNNInference(model_path, processor)
        self.methods.append(gnn)
    
    def generate_evidence_exact_training_config(self, model, query_node: str):
        """
        EXACT MATCH to training config:
        - 2 evidence nodes: 1 leaf + 1 intermediate
        - Evidence state: Always first state (0)
        """
        node_types = self.get_node_types(model)
        leaves = node_types["leaves"]
        intermediates = node_types["intermediates"]
        
        # Remove query node from candidates
        leaves = [n for n in leaves if n != query_node]
        intermediates = [n for n in intermediates if n != query_node]
        
        scenarios = []
        
        # Scenario 0: No evidence (baseline)
        scenarios.append({})
        
        # Scenario 1: Exactly as training - 1 leaf + 1 intermediate, state=0
        if len(leaves) >= 1 and len(intermediates) >= 1:
            evidence = {}
            
            # Pick first leaf
            leaf_node = leaves[0]
            cpd_leaf = model.get_cpds(leaf_node)
            if hasattr(cpd_leaf, 'state_names') and leaf_node in cpd_leaf.state_names:
                leaf_state = cpd_leaf.state_names[leaf_node][0]  # First state
                evidence[leaf_node] = leaf_state
            
            # Pick first intermediate
            int_node = intermediates[0]
            cpd_int = model.get_cpds(int_node)
            if hasattr(cpd_int, 'state_names') and int_node in cpd_int.state_names:
                int_state = cpd_int.state_names[int_node][0]  # First state
                evidence[int_node] = int_state
            
            if len(evidence) == 2:
                scenarios.append(evidence)
        
        # Scenario 2: Different pair (still 1 leaf + 1 intermediate, state=0)
        if len(leaves) >= 2 and len(intermediates) >= 2:
            evidence = {}
            
            leaf_node = leaves[1]
            cpd_leaf = model.get_cpds(leaf_node)
            if hasattr(cpd_leaf, 'state_names') and leaf_node in cpd_leaf.state_names:
                evidence[leaf_node] = cpd_leaf.state_names[leaf_node][0]
            
            int_node = intermediates[1]
            cpd_int = model.get_cpds(int_node)
            if hasattr(cpd_int, 'state_names') and int_node in cpd_int.state_names:
                evidence[int_node] = cpd_int.state_names[int_node][0]
            
            if len(evidence) == 2:
                scenarios.append(evidence)
        
        return scenarios if len(scenarios) > 1 else [{}]
    
    def get_node_types(self, model):
        """Helper to get node types"""
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
    
    def benchmark_network(self, bif_path: Path) -> Optional[Dict]:
        network_name = bif_path.stem
        print(f"\n{'='*60}")
        print(f"Benchmarking: {network_name}")
        print(f"{'='*60}")
        
        try:
            reader = BIFReader(str(bif_path))
            model = reader.get_model()
        except Exception as e:
            print(f"⚠ Failed to load: {e}")
            return None
        
        node_types = self.get_node_types(model)
        roots = node_types["roots"]
        
        if not roots:
            print("⚠ No root node found")
            return None
        
        query_node = roots[0]
        print(f"Query node: {query_node} (root)")
        print(f"Network: {model.number_of_nodes()} nodes, {model.number_of_edges()} edges")
        print(f"  Roots: {len(roots)}, Intermediates: {len(node_types['intermediates'])}, "
              f"Leaves: {len(node_types['leaves'])}")
        
        # Generate scenarios matching EXACT training config
        scenarios = self.generate_evidence_exact_training_config(model, query_node)
        print(f"\nTesting {len(scenarios)} scenarios (matching training config)")
        
        # Show evidence details
        for i, evidence in enumerate(scenarios):
            if evidence:
                ev_details = []
                for node, state in evidence.items():
                    node_type = "leaf" if node in node_types["leaves"] else \
                               "intermediate" if node in node_types["intermediates"] else "other"
                    ev_details.append(f"{node}({node_type})={state}")
                print(f"  Scenario {i}: {' AND '.join(ev_details)}")
            else:
                print(f"  Scenario {i}: No evidence (baseline)")
        
        # Ground truth
        gt_method = PgmpyVariableElimination()
        ground_truths = []
        for scenario in scenarios:
            gt_prob, _ = gt_method.infer(model, query_node, scenario)
            ground_truths.append(gt_prob)
        
        results = {
            'network_name': network_name,
            'num_nodes': model.number_of_nodes(),
            'num_edges': model.number_of_edges(),
            'query_node': query_node,
            'methods': {}
        }
        
        for method in self.methods:
            print(f"\n  Testing {method.name}...")
            method_results = {
                'predictions': [], 'times': [], 'errors': [], 'rel_errors': []
            }
            
            for i, (scenario, gt) in enumerate(zip(scenarios, ground_truths)):
                pred, elapsed = method.infer(model, query_node, scenario)
                error = abs(pred - gt)
                rel_error = error / (gt + 1e-8)
                
                method_results['predictions'].append(pred)
                method_results['times'].append(elapsed)
                method_results['errors'].append(error)
                method_results['rel_errors'].append(rel_error)
                
                print(f"    Scenario {i}: GT={gt:.4f}, Pred={pred:.4f}, "
                      f"Error={error:.4f}, Time={elapsed*1000:.2f}ms")
            
            method_results['mae'] = np.mean(method_results['errors'])
            method_results['rmse'] = np.sqrt(np.mean(np.array(method_results['errors'])**2))
            method_results['max_error'] = np.max(method_results['errors'])
            method_results['avg_time_ms'] = np.mean(method_results['times']) * 1000
            
            results['methods'][method.name] = method_results
            
            print(f"    → MAE: {method_results['mae']:.4f}, "
                  f"RMSE: {method_results['rmse']:.4f}, "
                  f"Time: {method_results['avg_time_ms']:.2f}ms")
        
        return results
    
    def run_full_benchmark(self, max_networks: Optional[int] = None) -> pd.DataFrame:
        all_results = []
        files = self.bif_files[:max_networks] if max_networks else self.bif_files
        
        for bif_path in files:
            result = self.benchmark_network(bif_path)
            if result:
                all_results.append(result)
        
        # Save
        results_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Summary
        summary_data = []
        for result in all_results:
            for method_name, method_data in result['methods'].items():
                summary_data.append({
                    'Network': result['network_name'],
                    'Nodes': result['num_nodes'],
                    'Edges': result['num_edges'],
                    'Method': method_name,
                    'MAE': method_data['mae'],
                    'RMSE': method_data['rmse'],
                    'Avg Time (ms)': method_data['avg_time_ms'],
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.output_dir, "summary.csv"), index=False)
        
        # Aggregate stats
        if len(df) > 0:
            print("\n" + "="*80)
            print("AGGREGATE STATISTICS")
            print("="*80)
            stats = df.groupby('Method').agg({
                'MAE': ['mean', 'std', 'min', 'max'],
                'Avg Time (ms)': ['mean', 'std', 'min', 'max']
            }).round(4)
            print(stats)
            stats.to_csv(os.path.join(self.output_dir, 'aggregate_stats.csv'))
        
        return df
    
    def visualize_results(self, df: pd.DataFrame):
        if len(df) == 0:
            return
        
        # MAE comparison
        plt.figure(figsize=(14, 6))
        pivot = df.pivot(index='Network', columns='Method', values='MAE')
        pivot.plot(kind='bar', figsize=(14, 6))
        plt.title('Mean Absolute Error (Training Config: 2 Evidence, State=0)')
        plt.ylabel('MAE')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mae_comparison.png'), dpi=300)
        plt.close()
        
        # Speed
        plt.figure(figsize=(14, 6))
        pivot_time = df.pivot(index='Network', columns='Method', values='Avg Time (ms)')
        pivot_time.plot(kind='bar', figsize=(14, 6), logy=True)
        plt.title('Inference Time (Log Scale)')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'time_comparison.png'), dpi=300)
        plt.close()
        
        print(f"✓ Visualizations saved to {self.output_dir}/")


def main():
    print("="*80)
    print("BENCHMARK: EXACT MATCH TO TRAINING CONFIGURATION")
    print("="*80)
    print("Evidence Configuration:")
    print("  - Number of evidence nodes: 2 (1 leaf + 1 intermediate)")
    print("  - Evidence state: Always first state (index 0)")
    print("  - Query: Root node, first state probability")
    print("="*80)
    
    bif_directory = "dataset_bif_files"
    output_dir = "benchmark_results_exact_config"
    model_path = "models/graphsage_root_probability_evidence_only_intermediate.pt"
    
    suite = BenchmarkSuite(bif_directory, output_dir)
    
    if os.path.exists(model_path):
        processor = BenchmarkDatasetProcessor(config_path="config.yaml")
        suite.add_gnn_method(model_path, processor)
    else:
        print(f"⚠ GNN model not found: {model_path}")
    
    df = suite.run_full_benchmark(max_networks=15)
    
    if len(df) > 0:
        suite.visualize_results(df)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()