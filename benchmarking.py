import os
import json
import time
import yaml
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from collections import defaultdict

from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.sampling import BayesianModelSampling

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ========== CONFIGURATION ==========
MAX_NODES_FILTER = None  # Set to limit network size (e.g., 500)
TIMEOUT_SECONDS = 30     # Timeout for slow methods

class InferenceMethod:
    """Base class for all inference methods"""
    def __init__(self, name: str, color: str = "blue"):
        self.name = name
        self.color = color
    
    def infer(self, model, query_node: str, evidence: dict) -> Tuple[float, float, bool]:
        """Returns: (probability, time_seconds, success)"""
        raise NotImplementedError

class PgmpyVariableElimination(InferenceMethod):
    """Gold standard: Exact inference"""
    def __init__(self):
        super().__init__("VE-Exact", color="#2E7D32")  # Green
    
    def infer(self, model, query_node: str, evidence: dict) -> Tuple[float, float, bool]:
        start = time.time()
        try:
            inference = VariableElimination(model)
            result = inference.query(
                variables=[query_node],
                evidence=evidence if evidence else None,
                show_progress=False
            )
            prob = float(result.values[0])
            elapsed = time.time() - start
            return prob, elapsed, True
        except Exception:
            elapsed = time.time() - start
            return 0.5, elapsed, False

class PgmpyBeliefPropagation(InferenceMethod):
    """Approximate inference: Faster but fails on loopy graphs"""
    def __init__(self, max_nodes: int = 500):
        super().__init__("BP-Approx", color="#D32F2F")  # Red
        self.max_nodes = max_nodes
    
    def infer(self, model, query_node: str, evidence: dict) -> Tuple[float, float, bool]:
        if model.number_of_nodes() > self.max_nodes:
            return 0.5, 0.0, False
        
        start = time.time()
        try:
            inference = BeliefPropagation(model)
            result = inference.query(
                variables=[query_node],
                evidence=evidence if evidence else None,
                show_progress=False
            )
            prob = float(result.values[0])
            elapsed = time.time() - start
            return prob, elapsed, True
        except Exception:
            elapsed = time.time() - start
            return 0.5, elapsed, False

class PgmpySampling(InferenceMethod):
    """Sampling-based: Scalable and reliable"""
    def __init__(self, n_samples: int = 10000):
        super().__init__(f"Sampling-{n_samples}", color="#1976D2")  # Blue
        self.n_samples = n_samples
    
    def infer(self, model, query_node: str, evidence: dict) -> Tuple[float, float, bool]:
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
            elapsed = time.time() - start
            return float(prob), elapsed, True
        except Exception:
            elapsed = time.time() - start
            return 0.5, elapsed, False


class UnifiedBenchmark:
    def __init__(self, bif_directory: str, output_dir: str = "benchmark_results_FIXED"):
        self.bif_directory = bif_directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.methods: List[InferenceMethod] = []
        self.bif_files = list(Path(bif_directory).glob("*.bif"))
        print(f"✅ Found {len(self.bif_files)} BIF files in {bif_directory}")

    def add_method(self, method: InferenceMethod):
        self.methods.append(method)
        print(f"✅ Added method: {method.name}")

    def generate_evidence_scenarios(self, model, query_node: str, max_scenarios: int = 3):
        """Generate evidence scenarios"""
        # Get node types
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
        
        leaves = [n for n in leaves if n != query_node]
        intermediates = [n for n in intermediates if n != query_node]
        
        scenarios = [{}]  # No evidence
        
        # Add evidence scenarios
        if len(leaves) >= 1 and len(intermediates) >= 1:
            for i in range(min(max_scenarios - 1, min(len(leaves), len(intermediates)))):
                evidence = {}
                
                # Add leaf
                leaf_node = leaves[i % len(leaves)]
                cpd_leaf = model.get_cpds(leaf_node)
                if hasattr(cpd_leaf, 'state_names') and leaf_node in cpd_leaf.state_names:
                    evidence[leaf_node] = cpd_leaf.state_names[leaf_node][0]
                
                # Add intermediate
                int_node = intermediates[i % len(intermediates)]
                cpd_int = model.get_cpds(int_node)
                if hasattr(cpd_int, 'state_names') and int_node in cpd_int.state_names:
                    evidence[int_node] = cpd_int.state_names[int_node][0]
                
                if len(evidence) == 2:
                    scenarios.append(evidence)
        
        return scenarios
    
    def benchmark_network(self, bif_path: Path) -> Optional[Dict]:
        """Benchmark a single network"""
        network_name = bif_path.stem
        print(f"\n{'='*70}")
        print(f"Network: {network_name}")
        print(f"{'='*70}")
        
        try:
            reader = BIFReader(str(bif_path))
            model = reader.get_model()
        except Exception as e:
            print(f"⚠ Failed to load: {e}")
            return None
        
        # Get root node
        roots = [n for n in model.nodes() if len(list(model.predecessors(n))) == 0]
        if not roots:
            print("⚠ No root node")
            return None
        
        query_node = roots[0]
        num_nodes = model.number_of_nodes()
        num_edges = model.number_of_edges()
        
        print(f"  Nodes: {num_nodes}, Edges: {num_edges}")
        print(f"  Query: {query_node} (root)")
        
        # Generate scenarios
        scenarios = self.generate_evidence_scenarios(model, query_node, max_scenarios=3)
        print(f"  Testing {len(scenarios)} scenarios")
        
        # Ground truth (Variable Elimination)
        gt_method = PgmpyVariableElimination()
        ground_truths = []
        for scenario in scenarios:
            gt_prob, gt_time, success = gt_method.infer(model, query_node, scenario)
            ground_truths.append(gt_prob)
            if not success:
                print(f"  ⚠ Ground truth failed for scenario")
        
        # Benchmark all methods
        results = {
            'network_name': network_name,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'query_node': query_node,
            'num_scenarios': len(scenarios),
            'ground_truths': ground_truths,  # ✅ STORE GROUND TRUTH
            'methods': {}
        }
        
        for method in self.methods:
            print(f"  Testing {method.name}...")
            method_results = {
                'predictions': [],
                'times': [],
                'errors': [],
                'successes': []
            }
            
            for i, (scenario, gt) in enumerate(zip(scenarios, ground_truths)):
                pred, elapsed, success = method.infer(model, query_node, scenario)
                error = abs(pred - gt) if success else float('inf')
                
                method_results['predictions'].append(pred if success else None)
                method_results['times'].append(elapsed)
                method_results['errors'].append(error)
                method_results['successes'].append(success)
            
            # ✅ COMPUTE AGGREGATE METRICS (DON'T HIDE FAILURES)
            valid_errors = [e for e in method_results['errors'] if e != float('inf')]
            if valid_errors:
                method_results['mae'] = np.mean(valid_errors)
                method_results['rmse'] = np.sqrt(np.mean(np.array(valid_errors)**2))
                method_results['max_error'] = np.max(valid_errors)
            else:
                method_results['mae'] = float('inf')
                method_results['rmse'] = float('inf')
                method_results['max_error'] = float('inf')
            
            method_results['avg_time_ms'] = np.mean(method_results['times']) * 1000
            method_results['success_rate'] = np.mean(method_results['successes'])
            
            results['methods'][method.name] = method_results
            
            print(f"    → MAE: {method_results['mae']:.4f}, "
                  f"Time: {method_results['avg_time_ms']:.2f}ms, "
                  f"Success: {method_results['success_rate']:.1%}")
        
        return results
  
    def run_full_benchmark(self, max_networks: Optional[int] = None) -> pd.DataFrame:
        print("\n" + "="*70)
        print("STARTING FULL BENCHMARK (FIXED VERSION)")
        print("="*70)
        
        all_results = []
        files = self.bif_files[:max_networks] if max_networks else self.bif_files
        
        for i, bif_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]", end=" ")
            result = self.benchmark_network(bif_path)
            if result:
                all_results.append(result)
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n✅ Saved detailed results to {results_path}")
        
        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            for method_name, method_data in result['methods'].items():
                summary_data.append({
                    'Network': result['network_name'],
                    'Nodes': result['num_nodes'],
                    'Edges': result['num_edges'],
                    'Method': method_name,
                    'MAE': method_data.get('mae', float('inf')),
                    'RMSE': method_data.get('rmse', float('inf')),
                    'Time_ms': method_data.get('avg_time_ms', 0),
                    'Success_Rate': method_data.get('success_rate', 0)
                })
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.output_dir, "summary.csv"), index=False)
        print(f"✅ Saved summary to {self.output_dir}/summary.csv")
        
        # ✅ COMPUTE CORRECTED AGGREGATE STATS
        self._compute_corrected_aggregate_stats(df)
        
        return df

    def _compute_corrected_aggregate_stats(self, df: pd.DataFrame):
        """✅ FIXED: Properly accounts for failures"""
        print("\n" + "="*70)
        print("CORRECTED AGGREGATE STATISTICS")
        print("="*70)
        
        results = {}
        
        for method in df['Method'].unique():
            method_df = df[df['Method'] == method]
            
            # Separate successful and failed attempts
            successful = method_df[method_df['MAE'] != float('inf')]
            failed = method_df[method_df['MAE'] == float('inf')]
            
            n_total = len(method_df)
            n_success = len(successful)
            n_failed = len(failed)
            success_rate = n_success / n_total if n_total > 0 else 0
            
            results[method] = {
                'success_rate': success_rate,
                'mae_when_successful': successful['MAE'].mean() if n_success > 0 else np.nan,
                'mae_std': successful['MAE'].std() if n_success > 0 else np.nan,
                'time_ms_median': successful['Time_ms'].median() if n_success > 0 else np.nan,
                # Combined metric (penalizing failures)
                'expected_mae': (success_rate * successful['MAE'].mean() + 
                               (1 - success_rate) * 0.5) if n_success > 0 else 0.5,
            }
        
        # Create formatted table
        df_results = pd.DataFrame(results).T
        
        print(f"\n{'Method':<20} {'Success':>8} {'MAE(✓)':>8} {'Time(ms)':>10} {'Expected':>10}")
        print(f"{'':20} {'Rate':>8} {'':>8} {'median':>10} {'MAE':>10}")
        print("-"*70)
        
        for method, data in results.items():
            print(f"{method:<20} {data['success_rate']:>7.1%} "
                  f"{data['mae_when_successful']:>8.4f} "
                  f"{data['time_ms_median']:>10.1f} "
                  f"{data['expected_mae']:>10.4f}")
        
        # Save
        df_results.to_csv(os.path.join(self.output_dir, 'corrected_aggregate_stats.csv'))
        print(f"\n✅ Saved corrected stats to {self.output_dir}/corrected_aggregate_stats.csv")
        
        # Identify failures
        print("\n" + "="*70)
        print("FAILURE ANALYSIS")
        print("="*70)
        
        for method in df['Method'].unique():
            method_df = df[df['Method'] == method]
            failed = method_df[method_df['MAE'] == float('inf')]
            
            if len(failed) > 0:
                print(f"\n❌ {method} failed on {len(failed)}/{len(method_df)} networks:")
                for _, row in failed.iterrows():
                    print(f"   - {row['Network']} ({row['Nodes']} nodes, {row['Edges']} edges)")
            else:
                print(f"\n✅ {method}: 100% success rate!")

    def visualize_results(self, df: pd.DataFrame):
        """✅ CORRECTED VISUALIZATIONS"""
        print("\n" + "="*70)
        print("GENERATING CORRECTED VISUALIZATIONS")
        print("="*70)
        
        # Create comprehensive 3-panel comparison
        self._create_comprehensive_comparison(df)
        self._create_network_by_network_plot(df)
        self._create_success_vs_accuracy_plot(df)
        
        print(f"\n✅ All visualizations saved to {self.output_dir}/")

    def _create_comprehensive_comparison(self, df: pd.DataFrame):
        """3-panel: Success Rate, Accuracy, True Performance"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        methods = sorted(df['Method'].unique())
        colors = {m: self._get_method_color(m) for m in methods}
        
        # Panel 1: Success Rate
        success_rates = []
        for method in methods:
            method_df = df[df['Method'] == method]
            successful = method_df[method_df['MAE'] != float('inf')]
            success_rates.append(100 * len(successful) / len(method_df))
        
        ax = axes[0]
        bars = ax.bar(range(len(methods)), success_rates, color=[colors[m] for m in methods], 
                     alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('-', '\n') for m in methods], fontsize=10)
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Reliability', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.axhline(100, color='green', linestyle='--', alpha=0.3, linewidth=2)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, success_rates):
            label = f'{val:.0f}%'
            ax.text(bar.get_x() + bar.get_width()/2., val + 2, label,
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Panel 2: MAE when successful
        mae_successful = []
        for method in methods:
            method_df = df[df['Method'] == method]
            successful = method_df[method_df['MAE'] != float('inf')]
            mae_successful.append(successful['MAE'].mean() if len(successful) > 0 else 0)
        
        ax = axes[1]
        bars = ax.bar(range(len(methods)), mae_successful, color=[colors[m] for m in methods],
                     alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('-', '\n') for m in methods], fontsize=10)
        ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy (When Works)', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(mae_successful) * 1.3 if max(mae_successful) > 0 else 0.1)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, mae_successful):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.003, f'{val:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Panel 3: Expected MAE (with failure penalty)
        expected_mae = []
        for method in methods:
            method_df = df[df['Method'] == method]
            successful = method_df[method_df['MAE'] != float('inf')]
            n_total = len(method_df)
            n_success = len(successful)
            success_rate = n_success / n_total
            mae_when_successful = successful['MAE'].mean() if n_success > 0 else 0
            expected = success_rate * mae_when_successful + (1 - success_rate) * 0.5
            expected_mae.append(expected)
        
        ax = axes[2]
        bars = ax.bar(range(len(methods)), expected_mae, color=[colors[m] for m in methods],
                     alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('-', '\n') for m in methods], fontsize=10)
        ax.set_ylabel('Expected MAE', fontsize=12, fontweight='bold')
        ax.set_title('TRUE Performance\n(with failure penalty)', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(expected_mae) * 1.3 if max(expected_mae) > 0 else 0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, expected_mae):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.015, f'{val:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: comprehensive_comparison.png")

    def _create_network_by_network_plot(self, df: pd.DataFrame):
        """Bar chart showing per-network performance"""
        fig, ax = plt.subplots(figsize=(16, 6))
        
        networks = sorted(df['Network'].unique())
        methods = sorted(df['Method'].unique())
        x = np.arange(len(networks))
        width = 0.25
        
        for i, method in enumerate(methods):
            maes = []
            for network in networks:
                network_data = df[(df['Network'] == network) & (df['Method'] == method)]
                mae = network_data['MAE'].values[0]
                if mae == float('inf'):
                    mae = 0.4  # Cap for visualization
                maes.append(mae)
            
            offset = width * (i - 1)
            bars = ax.bar(x + offset, maes, width, label=method, 
                         color=self._get_method_color(method), alpha=0.8, edgecolor='black')
            
            # Mark failures
            for j, (bar, mae_val) in enumerate(zip(bars, maes)):
                if mae_val == 0.4:  # Was inf
                    ax.text(bar.get_x() + bar.get_width()/2., 0.2, 'FAIL',
                           ha='center', va='center', fontweight='bold', 
                           fontsize=7, color='white', rotation=90)
        
        ax.set_xlabel('Bayesian Network', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax.set_title('Per-Network Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(networks, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=11)
        ax.set_ylim(0, 0.45)
        ax.axhline(0.4, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'network_by_network.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: network_by_network.png")

    def _create_success_vs_accuracy_plot(self, df: pd.DataFrame):
        """Scatter: Success Rate vs MAE"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        methods = sorted(df['Method'].unique())
        
        for method in methods:
            method_df = df[df['Method'] == method]
            successful = method_df[method_df['MAE'] != float('inf')]
            
            n_total = len(method_df)
            n_success = len(successful)
            success_rate = 100 * n_success / n_total
            mae = successful['MAE'].mean() if n_success > 0 else 0
            
            median_time = successful['Time_ms'].median() if n_success > 0 else 1000
            size = min(2000, max(200, 5000 / np.log10(median_time + 1)))
            
            ax.scatter(success_rate, mae, s=size, alpha=0.7, 
                      color=self._get_method_color(method),
                      edgecolors='black', linewidths=2, label=method)
            
            ax.text(success_rate, mae + 0.01, method.replace('-', ' '),
                   ha='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Success Rate (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('MAE (when successful)', fontsize=13, fontweight='bold')
        ax.set_title('Success vs Accuracy Trade-off', fontsize=14, fontweight='bold')
        ax.set_xlim(20, 105)
        ax.axvline(100, color='green', linestyle='--', alpha=0.3, linewidth=2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'success_vs_accuracy.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: success_vs_accuracy.png")

    def _get_method_color(self, method_name: str) -> str:
        """Get color for method"""
        if 'VE-Exact' in method_name:
            return '#2E7D32'  # Green
        elif 'BP-Approx' in method_name:
            return '#D32F2F'  # Red
        elif 'Sampling' in method_name:
            return '#1976D2'  # Blue
        else:
            return '#757575'  # Gray


def main():
    print("="*80)
    print("FIXED UNIFIED BENCHMARK: Traditional Inference Methods")
    print("="*80)
    
    # Configuration
    bif_directory = "dataset_bif_files"  # UPDATE THIS PATH
    output_dir = "benchmark_results_FIXED"
    
    # Create benchmark suite
    suite = UnifiedBenchmark(bif_directory, output_dir)
    
    # Add methods
    print("\nAdding inference methods:")
    suite.add_method(PgmpyVariableElimination())
    suite.add_method(PgmpyBeliefPropagation(max_nodes=500))
    suite.add_method(PgmpySampling(n_samples=10000))
    
    # Run benchmark
    df = suite.run_full_benchmark(max_networks=None)
    
    # Visualize
    if len(df) > 0:
        suite.visualize_results(df)
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}/")
    print("\nKey files:")
    print(f"  - detailed_results.json: Complete per-scenario results")
    print(f"  - summary.csv: Flat table for analysis")
    print(f"  - corrected_aggregate_stats.csv: Fixed statistics")
    print(f"  - comprehensive_comparison.png: 3-panel comparison")
    print(f"  - network_by_network.png: Per-network breakdown")
    print(f"  - success_vs_accuracy.png: Trade-off visualization")

if __name__ == "__main__":
    main()