"""
PROPER BENCHMARKING COMPARISON TOOL
====================================
Compares GNN models with traditional inference methods
Handles different JSON formats properly!

Supports:
- GNN models: GraphSAGE, GCN, GAT (single scenario per network)
- Traditional methods: VE, BP, Sampling (multiple scenarios, can fail)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
from scipy import stats
from typing import Dict, List, Tuple

# Publication settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class ProperBenchmarkComparison:
    """Handles both GNN and traditional method formats"""
    
    def __init__(self, output_dir="proper_benchmark_comparison"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/tables", exist_ok=True)
        
        # Store results
        self.gnn_results = {}  # GNN models
        self.trad_results = {}  # Traditional methods
        
        # Colors
        self.colors = {
            'graphsage': '#2ecc71',     # Green
            'gcn': '#3498db',           # Blue  
            'gat': '#e74c3c',           # Red
            've_exact': '#9b59b6',      # Purple
            'bp_approx': '#f39c12',     # Orange
            'sampling': '#1abc9c',      # Teal
        }
        
        self.display_names = {
            'graphsage': 'GraphSAGE (GNN)',
            'gcn': 'GCN (GNN)',
            'gat': 'GAT (GNN)',
            've_exact': 'Variable Elimination',
            'bp_approx': 'Belief Propagation',
            'sampling': 'Monte Carlo Sampling',
        }
    
    def load_gnn_results(self, method_name: str, json_file: str):
        """Load GNN benchmark results (single scenario per network)"""
        print(f"\n📊 Loading GNN: {method_name.upper()}")
        
        if not os.path.exists(json_file):
            print(f"  ❌ File not found: {json_file}")
            return False
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Parse GNN format
        agg = data['aggregate_metrics']
        
        self.gnn_results[method_name] = {
            'type': 'gnn',
            'aggregate': {
                'mae': agg.get('mae', np.nan),
                'rmse': agg.get('rmse', np.nan),
                'r2_score': agg.get('r2_score', np.nan),
                'avg_time_ms': agg.get('avg_time_ms', np.nan),
                'accuracy_10pct': agg.get('accuracy_within_10pct', np.nan),
                'high_risk_mae': agg.get('high_risk_mae', np.nan),
                'success_rate': 1.0,  # GNNs always succeed
            },
            'per_network': {}
        }
        
        # Per-network results
        for net in data['per_network_results']:
            net_name = net['network_name']
            self.gnn_results[method_name]['per_network'][net_name] = {
                'mae': net['absolute_error'],
                'time_ms': net['inference_time_ms'],
                'success': True,
                'num_nodes': net['num_nodes'],
                'num_edges': net['num_edges']
            }
        
        print(f"  ✅ MAE: {agg.get('mae', 0):.4f}, "
              f"R²: {agg.get('r2_score', 0):.4f}, "
              f"Time: {agg.get('avg_time_ms', 0):.2f}ms, "
              f"Networks: {len(data['per_network_results'])}")
        
        return True
    
    def load_traditional_results(self, json_file: str):
        """Load traditional method results (from unified_benchmark_FIXED.py)"""
        print(f"\n📊 Loading Traditional Methods from: {json_file}")
        
        if not os.path.exists(json_file):
            print(f"  ❌ File not found: {json_file}")
            return False
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Parse each method from the detailed_results.json
        method_aggregates = {}
        method_per_network = {}
        
        for network_result in data:
            net_name = network_result['network_name']
            num_nodes = network_result['num_nodes']
            num_edges = network_result['num_edges']
            
            for method_name, method_data in network_result['methods'].items():
                # Normalize method names
                method_key = method_name.lower().replace('-', '_').replace(' ', '_')
                
                if method_key not in method_aggregates:
                    method_aggregates[method_key] = {
                        'maes': [],
                        'times': [],
                        'successes': [],
                    }
                    method_per_network[method_key] = {}
                
                # Average over scenarios for this network
                mae = method_data.get('mae', np.nan)
                time_ms = method_data.get('avg_time_ms', np.nan)
                success_rate = method_data.get('success_rate', 0.0)
                
                method_aggregates[method_key]['maes'].append(mae)
                method_aggregates[method_key]['times'].append(time_ms)
                method_aggregates[method_key]['successes'].append(success_rate)
                
                method_per_network[method_key][net_name] = {
                    'mae': mae if mae != float('inf') else np.nan,
                    'time_ms': time_ms,
                    'success': success_rate >= 0.5,  # Consider success if >=50% scenarios work
                    'success_rate': success_rate,
                    'num_nodes': num_nodes,
                    'num_edges': num_edges
                }
        
        # Compute aggregates for each method
        for method_key, agg_data in method_aggregates.items():
            # Filter out failures (inf MAE)
            valid_maes = [m for m in agg_data['maes'] if m != float('inf') and not np.isnan(m)]
            valid_times = [t for t in agg_data['times'] if not np.isnan(t)]
            
            overall_success_rate = np.mean(agg_data['successes'])
            
            self.trad_results[method_key] = {
                'type': 'traditional',
                'aggregate': {
                    'mae': np.mean(valid_maes) if valid_maes else np.nan,
                    'rmse': np.sqrt(np.mean(np.array(valid_maes)**2)) if valid_maes else np.nan,
                    'r2_score': np.nan,  # Not computed for traditional methods
                    'avg_time_ms': np.mean(valid_times) if valid_times else np.nan,
                    'accuracy_10pct': np.nan,  # Not computed
                    'high_risk_mae': np.nan,  # Not computed
                    'success_rate': overall_success_rate,
                },
                'per_network': method_per_network[method_key]
            }
            
            print(f"  ✅ {self.display_names.get(method_key, method_key)}: "
                  f"MAE: {self.trad_results[method_key]['aggregate']['mae']:.4f}, "
                  f"Time: {self.trad_results[method_key]['aggregate']['avg_time_ms']:.2f}ms, "
                  f"Success: {overall_success_rate:.1%}")
        
        return True
    
    def create_overall_comparison_table(self):
        """Create summary table"""
        print("\n" + "="*100)
        print("OVERALL PERFORMANCE COMPARISON")
        print("="*100)
        
        rows = []
        
        # GNN methods
        for method_name, data in self.gnn_results.items():
            agg = data['aggregate']
            rows.append({
                'Method': self.display_names.get(method_name, method_name),
                'Type': 'GNN',
                'MAE ↓': f"{agg['mae']:.4f}",
                'R² ↑': f"{agg['r2_score']:.4f}",
                'Time (ms) ↓': f"{agg['avg_time_ms']:.2f}",
                'Success Rate': '100%',
                'High-Risk MAE': f"{agg.get('high_risk_mae', np.nan):.4f}"
            })
        
        # Traditional methods
        for method_name, data in self.trad_results.items():
            agg = data['aggregate']
            rows.append({
                'Method': self.display_names.get(method_name, method_name),
                'Type': 'Traditional',
                'MAE ↓': f"{agg['mae']:.4f}" if not np.isnan(agg['mae']) else "N/A",
                'R² ↑': "N/A",
                'Time (ms) ↓': f"{agg['avg_time_ms']:.2f}",
                'Success Rate': f"{agg['success_rate']:.1%}",
                'High-Risk MAE': "N/A"
            })
        
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        
        # Save
        csv_path = f"{self.output_dir}/tables/overall_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved to {csv_path}")
        
        return df
    
    def plot_speed_vs_accuracy_scatter(self):
        """THE KEY PLOT: Speed vs Accuracy trade-off"""
        print("\n📊 Creating Speed vs Accuracy plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        all_methods = {**self.gnn_results, **self.trad_results}
        
        for method_name, data in all_methods.items():
            agg = data['aggregate']
            
            time_ms = agg['avg_time_ms']
            mae = agg['mae']
            success_rate = agg['success_rate']
            
            if np.isnan(time_ms) or np.isnan(mae):
                continue
            
            # Adjust MAE for failures (penalty)
            effective_mae = mae if success_rate >= 0.95 else mae / success_rate
            
            color = self.colors.get(method_name, '#95a5a6')
            label = self.display_names.get(method_name, method_name)
            
            # Size based on success rate
            size = 400 * success_rate if data['type'] == 'traditional' else 400
            
            # Plot
            marker = 'o' if data['type'] == 'gnn' else '^'
            ax.scatter(time_ms, effective_mae, s=size, c=color, alpha=0.7,
                      edgecolors='black', linewidths=2, marker=marker, 
                      label=label, zorder=3)
            
            # Label
            ax.text(time_ms, effective_mae + 0.01, label.split()[0],
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Ideal zones
        ax.axhspan(0, 0.05, alpha=0.1, color='green', zorder=1)
        ax.text(0.98, 0.025, 'HIGH ACCURACY', transform=ax.transData,
               ha='right', va='center', fontsize=10, color='green',
               fontweight='bold', alpha=0.7)
        
        ax.axvspan(0, 2, alpha=0.1, color='blue', zorder=1)
        ax.text(1, 0.18, 'FAST', rotation=90, va='center', ha='center',
               fontsize=10, color='blue', fontweight='bold', alpha=0.7)
        
        ax.set_xlabel('Inference Time (ms)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error (lower = better)', fontsize=13, fontweight='bold')
        ax.set_title('Speed vs Accuracy Trade-off\n(GNN Methods: Circle, Traditional: Triangle)', 
                    fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        save_path = f"{self.output_dir}/plots/speed_vs_accuracy.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_comprehensive_comparison(self):
        """4-panel comparison: MAE, Speed, Success Rate, Scalability"""
        print("\n📊 Creating comprehensive comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        all_methods = {**self.gnn_results, **self.trad_results}
        method_names = list(all_methods.keys())
        
        # Panel 1: MAE Comparison
        ax = axes[0, 0]
        maes = []
        colors_list = []
        labels = []
        
        for method in method_names:
            mae = all_methods[method]['aggregate']['mae']
            if not np.isnan(mae):
                maes.append(mae)
                colors_list.append(self.colors.get(method, '#95a5a6'))
                labels.append(self.display_names.get(method, method))
        
        bars = ax.bar(range(len(labels)), maes, color=colors_list,
                     alpha=0.8, edgecolor='black', linewidth=2)
        
        # Highlight best
        best_idx = np.argmin(maes)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(4)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels], 
                          fontsize=9, rotation=0)
        ax.set_ylabel('Mean Absolute Error', fontweight='bold')
        ax.set_title('Accuracy (Lower is Better)', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, maes)):
            ax.text(i, val + 0.005, f'{val:.4f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
        
        # Panel 2: Speed Comparison
        ax = axes[0, 1]
        times = []
        colors_list = []
        labels = []
        
        for method in method_names:
            time_ms = all_methods[method]['aggregate']['avg_time_ms']
            if not np.isnan(time_ms):
                times.append(time_ms)
                colors_list.append(self.colors.get(method, '#95a5a6'))
                labels.append(self.display_names.get(method, method))
        
        bars = ax.bar(range(len(labels)), times, color=colors_list,
                     alpha=0.8, edgecolor='black', linewidth=2)
        
        # Highlight fastest
        fastest_idx = np.argmin(times)
        bars[fastest_idx].set_edgecolor('gold')
        bars[fastest_idx].set_linewidth(4)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels],
                          fontsize=9, rotation=0)
        ax.set_ylabel('Inference Time (ms)', fontweight='bold')
        ax.set_title('Speed (Lower is Better)', fontweight='bold', fontsize=13)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, times)):
            ax.text(i, val * 1.2, f'{val:.2f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
        
        # Panel 3: Success Rate
        ax = axes[1, 0]
        success_rates = []
        colors_list = []
        labels = []
        
        for method in method_names:
            sr = all_methods[method]['aggregate']['success_rate'] * 100
            success_rates.append(sr)
            colors_list.append(self.colors.get(method, '#95a5a6'))
            labels.append(self.display_names.get(method, method))
        
        bars = ax.bar(range(len(labels)), success_rates, color=colors_list,
                     alpha=0.8, edgecolor='black', linewidth=2)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels],
                          fontsize=9, rotation=0)
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_title('Reliability (Higher is Better)', fontweight='bold', fontsize=13)
        ax.set_ylim(0, 105)
        ax.axhline(100, color='green', linestyle='--', alpha=0.5, linewidth=2)
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, success_rates)):
            ax.text(i, val + 2, f'{val:.0f}%', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
        
        # Panel 4: Scalability (Time vs Network Size)
        ax = axes[1, 1]
        
        for method in method_names:
            per_net = all_methods[method]['per_network']
            
            sizes = []
            times_per_net = []
            
            for net_name, net_data in per_net.items():
                if net_data.get('success', True):
                    sizes.append(net_data['num_nodes'])
                    times_per_net.append(net_data['time_ms'])
            
            if len(sizes) > 5:
                # Sort by size
                sorted_indices = np.argsort(sizes)
                sizes = np.array(sizes)[sorted_indices]
                times_per_net = np.array(times_per_net)[sorted_indices]
                
                color = self.colors.get(method, '#95a5a6')
                label = self.display_names.get(method, method)
                
                ax.scatter(sizes, times_per_net, alpha=0.5, s=30, c=color)
                
                # Fit trend line
                if len(sizes) > 10:
                    z = np.polyfit(np.log10(sizes + 1), np.log10(times_per_net + 1), 1)
                    p = np.poly1d(z)
                    x_trend = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 50)
                    y_trend = 10 ** p(np.log10(x_trend + 1))
                    ax.plot(x_trend, y_trend, '--', color=color, linewidth=2,
                           alpha=0.8, label=label)
        
        ax.set_xlabel('Network Size (# nodes)', fontweight='bold')
        ax.set_ylabel('Inference Time (ms)', fontweight='bold')
        ax.set_title('Scalability Analysis', fontweight='bold', fontsize=13)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)
        
        plt.suptitle('Comprehensive Benchmarking Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = f"{self.output_dir}/plots/comprehensive_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_per_network_heatmap(self):
        """Heatmap showing performance across networks"""
        print("\n📊 Creating per-network heatmap...")
        
        # Get all network names
        all_networks = set()
        all_methods = {**self.gnn_results, **self.trad_results}
        
        for method_data in all_methods.values():
            all_networks.update(method_data['per_network'].keys())
        
        networks = sorted(list(all_networks))[:20]  # Top 20 networks
        methods = list(all_methods.keys())
        
        # Create MAE matrix
        mae_matrix = np.zeros((len(methods), len(networks)))
        
        for i, method in enumerate(methods):
            for j, network in enumerate(networks):
                net_data = all_methods[method]['per_network'].get(network, {})
                mae = net_data.get('mae', np.nan)
                if net_data.get('success', True) and not np.isnan(mae):
                    mae_matrix[i, j] = mae
                else:
                    mae_matrix[i, j] = np.nan  # Failed
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Mask NaN values
        masked_matrix = np.ma.masked_where(np.isnan(mae_matrix), mae_matrix)
        
        im = ax.imshow(masked_matrix, cmap='RdYlGn_r', aspect='auto',
                      vmin=0, vmax=0.2, interpolation='nearest')
        
        ax.set_xticks(range(len(networks)))
        ax.set_xticklabels(networks, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([self.display_names.get(m, m) for m in methods],
                          fontsize=10)
        
        ax.set_title('Per-Network MAE Comparison (White = Failed)', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Absolute Error', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(networks)):
                if not np.isnan(mae_matrix[i, j]):
                    text = ax.text(j, i, f'{mae_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", 
                                 fontsize=7, fontweight='bold')
        
        plt.tight_layout()
        save_path = f"{self.output_dir}/plots/per_network_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_gnn_advantage_summary(self):
        """Highlight GNN advantages vs traditional methods"""
        print("\n📊 Creating GNN advantage summary...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Compute averages
        gnn_mae_avg = np.mean([d['aggregate']['mae'] for d in self.gnn_results.values()])
        gnn_time_avg = np.mean([d['aggregate']['avg_time_ms'] for d in self.gnn_results.values()])
        gnn_success = 1.0
        
        trad_mae_avg = np.mean([d['aggregate']['mae'] for d in self.trad_results.values() 
                                if not np.isnan(d['aggregate']['mae'])])
        trad_time_avg = np.mean([d['aggregate']['avg_time_ms'] for d in self.trad_results.values()
                                 if not np.isnan(d['aggregate']['avg_time_ms'])])
        trad_success = np.mean([d['aggregate']['success_rate'] for d in self.trad_results.values()])
        
        # Panel 1: Accuracy
        ax = axes[0]
        bars = ax.bar(['GNN\nMethods', 'Traditional\nMethods'], 
                     [gnn_mae_avg, trad_mae_avg],
                     color=['#2ecc71', '#e74c3c'], alpha=0.8,
                     edgecolor='black', linewidth=2)
        ax.set_ylabel('Average MAE', fontweight='bold', fontsize=12)
        ax.set_title('Accuracy Comparison', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, [gnn_mae_avg, trad_mae_avg]):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                   f'{val:.4f}', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
        
        # Panel 2: Speed
        ax = axes[1]
        speedup = trad_time_avg / gnn_time_avg
        bars = ax.bar(['GNN\nMethods', 'Traditional\nMethods'],
                     [gnn_time_avg, trad_time_avg],
                     color=['#2ecc71', '#e74c3c'], alpha=0.8,
                     edgecolor='black', linewidth=2)
        ax.set_ylabel('Average Time (ms)', fontweight='bold', fontsize=12)
        ax.set_title(f'Speed Comparison\n(GNNs are {speedup:.1f}x faster)', 
                    fontweight='bold', fontsize=14)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, [gnn_time_avg, trad_time_avg]):
            ax.text(bar.get_x() + bar.get_width()/2., val * 1.5,
                   f'{val:.2f}ms', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
        
        # Panel 3: Reliability
        ax = axes[2]
        bars = ax.bar(['GNN\nMethods', 'Traditional\nMethods'],
                     [gnn_success * 100, trad_success * 100],
                     color=['#2ecc71', '#e74c3c'], alpha=0.8,
                     edgecolor='black', linewidth=2)
        ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=12)
        ax.set_title('Reliability', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 105)
        ax.axhline(100, color='green', linestyle='--', alpha=0.5, linewidth=2)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, [gnn_success * 100, trad_success * 100]):
            ax.text(bar.get_x() + bar.get_width()/2., val + 2,
                   f'{val:.1f}%', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
        
        plt.suptitle('GNN Advantages Over Traditional Methods',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = f"{self.output_dir}/plots/gnn_advantage_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def generate_summary_report(self):
        """Generate text report"""
        print("\n📝 Generating summary report...")
        
        report_path = f"{self.output_dir}/BENCHMARK_REPORT.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE BENCHMARKING REPORT\n")
            f.write("GNN Models vs Traditional Inference Methods\n")
            f.write("="*80 + "\n\n")
            
            # GNN Results
            f.write("GNN MODELS\n")
            f.write("-"*80 + "\n")
            for method_name, data in self.gnn_results.items():
                agg = data['aggregate']
                f.write(f"\n{self.display_names.get(method_name, method_name)}:\n")
                f.write(f"  MAE:  {agg['mae']:.4f}\n")
                f.write(f"  R²:   {agg['r2_score']:.4f}\n")
                f.write(f"  Time: {agg['avg_time_ms']:.2f} ms\n")
                f.write(f"  Success: 100%\n")
            
            # Traditional Methods
            f.write("\n\nTRADITIONAL METHODS\n")
            f.write("-"*80 + "\n")
            for method_name, data in self.trad_results.items():
                agg = data['aggregate']
                f.write(f"\n{self.display_names.get(method_name, method_name)}:\n")
                f.write(f"  MAE:  {agg['mae']:.4f}\n")
                f.write(f"  Time: {agg['avg_time_ms']:.2f} ms\n")
                f.write(f"  Success: {agg['success_rate']:.1%}\n")
            
            # Key Findings
            f.write("\n\n" + "="*80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*80 + "\n\n")
            
            # Best GNN
            if self.gnn_results:
                best_gnn = min(self.gnn_results.items(), 
                              key=lambda x: x[1]['aggregate']['mae'])
                f.write(f"Best GNN Model: {self.display_names.get(best_gnn[0], best_gnn[0])}\n")
                f.write(f"  MAE: {best_gnn[1]['aggregate']['mae']:.4f}\n")
                f.write(f"  R²: {best_gnn[1]['aggregate']['r2_score']:.4f}\n\n")
            
            # Speed advantage
            if self.gnn_results and self.trad_results:
                gnn_time_avg = np.mean([d['aggregate']['avg_time_ms'] 
                                       for d in self.gnn_results.values()])
                trad_time_avg = np.mean([d['aggregate']['avg_time_ms'] 
                                        for d in self.trad_results.values()
                                        if not np.isnan(d['aggregate']['avg_time_ms'])])
                speedup = trad_time_avg / gnn_time_avg
                
                f.write(f"Speed Advantage: GNNs are {speedup:.1f}x faster\n")
                f.write(f"  GNN avg: {gnn_time_avg:.2f} ms\n")
                f.write(f"  Traditional avg: {trad_time_avg:.2f} ms\n\n")
            
            f.write("\nGenerated Visualizations:\n")
            f.write("  - speed_vs_accuracy.png\n")
            f.write("  - comprehensive_comparison.png\n")
            f.write("  - per_network_heatmap.png\n")
            f.write("  - gnn_advantage_summary.png\n")
        
        print(f"  ✅ Report saved to {report_path}")


def main():
    """Main comparison pipeline"""
    print("="*100)
    print("PROPER BENCHMARKING COMPARISON PIPELINE")
    print("="*100)
    
    comp = ProperBenchmarkComparison(output_dir="proper_benchmark_comparison")
    
    # ============================================
    # LOAD GNN RESULTS
    # ============================================
    print("\n" + "="*100)
    print("LOADING GNN MODELS")
    print("="*100)
    
    comp.load_gnn_results('graphsage', 'graphsage_results.json')
    comp.load_gnn_results('gcn', 'gcn_results.json')
    comp.load_gnn_results('gat', 'gat_results.json')
    
    # ============================================
    # LOAD TRADITIONAL METHODS
    # ============================================
    print("\n" + "="*100)
    print("LOADING TRADITIONAL METHODS")
    print("="*100)
    
    # Load from unified_benchmark_FIXED.py output
    comp.load_traditional_results('benchmark_results_FIXED/detailed_results.json')
    
    # ============================================
    # GENERATE ALL COMPARISONS
    # ============================================
    print("\n" + "="*100)
    print("GENERATING COMPARISONS")
    print("="*100)
    
    # Tables
    comp.create_overall_comparison_table()
    
    # Plots
    comp.plot_speed_vs_accuracy_scatter()
    comp.plot_comprehensive_comparison()
    comp.plot_per_network_heatmap()
    comp.plot_gnn_advantage_summary()
    
    # Report
    comp.generate_summary_report()
    
    print("\n" + "="*100)
    print("✅ COMPARISON COMPLETE!")
    print("="*100)
    print(f"\nAll results saved to: proper_benchmark_comparison/")
    print("  - Tables: proper_benchmark_comparison/tables/")
    print("  - Plots: proper_benchmark_comparison/plots/")
    print("  - Report: proper_benchmark_comparison/BENCHMARK_REPORT.txt")


if __name__ == "__main__":
    main()