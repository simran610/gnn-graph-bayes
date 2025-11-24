# import os
# import json
# import matplotlib.pyplot as plt
# import numpy as np
# from collections import Counter, defaultdict
# from tqdm import tqdm
# import seaborn as sns

# # ========= Configuration =========
# FOLDER = "generated_graphs"
# NUM_SAMPLE_VIS = 20  # Number of sample graphs to visualize
# SAVE_PLOTS = True
# OUTPUT_DIR = "analysis_results"
# # =================================

# def load_and_analyze_graph(file_path):
#     """Efficiently loads and analyzes a single graph in one pass."""
#     try:
#         with open(file_path, "r") as f:
#             data = json.load(f)
        
#         nodes = data.get("nodes", {})
#         edges = data.get("edges", [])
        
#         num_nodes = len(nodes)
#         num_edges = len(edges)
        
#         # Count node types and collect root priors in one pass
#         roots = 0
#         leaves = 0
#         intermediates = 0
#         root_priors = []
        
#         for node_id, node in nodes.items():
#             node_type = node.get("type", "unknown")
            
#             if node_type == "root":
#                 roots += 1
#                 # Extract root node prior probability
#                 cpd = node.get("cpd", {})
#                 if "values" in cpd:
#                     try:
#                         values = cpd["values"]
#                         variable_card = cpd.get("variable_card", 2)
                        
#                         # For root nodes, values is a flat list: [P(State=0), P(State=1), ...]
#                         # We want P(State=1) which is at index 1
#                         if isinstance(values, list) and len(values) >= variable_card and variable_card >= 2:
#                             prior_prob = values[1]  # P(State=1)
#                             root_priors.append(float(prior_prob))
#                     except (IndexError, KeyError, TypeError, ValueError) as e:
#                         pass
#             elif node_type == "leaf":
#                 leaves += 1
#             elif node_type == "intermediate":
#                 intermediates += 1
        
#         # Calculate density
#         density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
#         return {
#             "nodes": num_nodes,
#             "edges": num_edges,
#             "density": density,
#             "roots": roots,
#             "leaves": leaves,
#             "intermediates": intermediates,
#             "root_priors": root_priors,
#             "file": os.path.basename(file_path)
#         }
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return None

# def main():
#     """Main analysis pipeline."""
    
#     # Create output directory
#     if SAVE_PLOTS and not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)
    
#     # 1. Find all graph files
#     print("🔍 Scanning for graph files...")
#     all_files = [
#         os.path.join(FOLDER, f)
#         for f in os.listdir(FOLDER)
#         if f.startswith("detailed_graph_") and f.endswith(".json")
#     ]
#     all_files.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
    
#     if not all_files:
#         print(f"❌ No graph files found in '{FOLDER}'")
#         return
    
#     print(f"📂 Found {len(all_files)} graph files\n")
    
#     # 2. Analyze all graphs efficiently
#     print("📊 Analyzing graphs...")
#     results = []
#     for fpath in tqdm(all_files, desc="Processing"):
#         result = load_and_analyze_graph(fpath)
#         if result:
#             results.append(result)
    
#     # 3. Extract arrays for analysis
#     num_nodes = np.array([r["nodes"] for r in results])
#     num_edges = np.array([r["edges"] for r in results])
#     densities = np.array([r["density"] for r in results])
#     num_roots = np.array([r["roots"] for r in results])
#     num_leaves = np.array([r["leaves"] for r in results])
#     num_intermediates = np.array([r["intermediates"] for r in results])
    
#     # Flatten all root priors
#     all_root_priors = [p for r in results for p in r["root_priors"]]
    
#     # 4. Print Summary Statistics
#     print("\n" + "="*60)
#     print("📈 DATASET SUMMARY STATISTICS")
#     print("="*60)
#     print(f"Total Graphs Analyzed: {len(results)}")
#     print(f"\n--- STRUCTURAL PROPERTIES ---")
#     print(f"Nodes per Graph:")
#     print(f"  Mean: {num_nodes.mean():.2f} | Median: {np.median(num_nodes):.0f}")
#     print(f"  Min: {num_nodes.min()} | Max: {num_nodes.max()}")
#     print(f"  Std Dev: {num_nodes.std():.2f}")
    
#     print(f"\nEdges per Graph:")
#     print(f"  Mean: {num_edges.mean():.2f} | Median: {np.median(num_edges):.0f}")
#     print(f"  Min: {num_edges.min()} | Max: {num_edges.max()}")
#     print(f"  Std Dev: {num_edges.std():.2f}")
    
#     print(f"\nGraph Density:")
#     print(f"  Mean: {densities.mean():.4f} | Median: {np.median(densities):.4f}")
#     print(f"  Min: {densities.min():.4f} | Max: {densities.max():.4f}")
    
#     print(f"\n--- NODE TYPE DISTRIBUTION ---")
#     print(f"Root Nodes per Graph:")
#     print(f"  Mean: {num_roots.mean():.2f} | Median: {np.median(num_roots):.0f}")
#     print(f"  Min: {num_roots.min()} | Max: {num_roots.max()}")
    
#     print(f"\nIntermediate Nodes per Graph:")
#     print(f"  Mean: {num_intermediates.mean():.2f} | Median: {np.median(num_intermediates):.0f}")
#     print(f"  Min: {num_intermediates.min()} | Max: {num_intermediates.max()}")
    
#     print(f"\nLeaf Nodes per Graph:")
#     print(f"  Mean: {num_leaves.mean():.2f} | Median: {np.median(num_leaves):.0f}")
#     print(f"  Min: {num_leaves.min()} | Max: {num_leaves.max()}")
    
#     print(f"\n--- ROOT NODE PRIOR PROBABILITIES ---")
#     print(f"Total Root Nodes Analyzed: {len(all_root_priors)}")
#     if all_root_priors:
#         print(f"Prior P(State=1):")
#         print(f"  Mean: {np.mean(all_root_priors):.4f} | Median: {np.median(all_root_priors):.4f}")
#         print(f"  Min: {np.min(all_root_priors):.4f} | Max: {np.max(all_root_priors):.4f}")
#         print(f"  Std Dev: {np.std(all_root_priors):.4f}")
        
#         # Analysis of low probability roots
#         low_prob_count = sum(1 for p in all_root_priors if p < 0.3)
#         mid_prob_count = sum(1 for p in all_root_priors if 0.3 <= p <= 0.7)
#         high_prob_count = sum(1 for p in all_root_priors if p > 0.7)
        
#         print(f"\n  Low Probability (<0.3): {low_prob_count} ({100*low_prob_count/len(all_root_priors):.1f}%)")
#         print(f"  Mid Probability (0.3-0.7): {mid_prob_count} ({100*mid_prob_count/len(all_root_priors):.1f}%)")
#         print(f"  High Probability (>0.7): {high_prob_count} ({100*high_prob_count/len(all_root_priors):.1f}%)")
    
#     print("="*60 + "\n")
    
#     # 5. Create Comprehensive Visualizations
#     print("🎨 Creating visualizations...")
    
#     # Set style
#     sns.set_style("whitegrid")
    
#     # Figure 1: Main Distributions (2x3 grid)
#     fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
    
#     # Node count distribution
#     axes[0, 0].hist(num_nodes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
#     axes[0, 0].axvline(num_nodes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {num_nodes.mean():.1f}')
#     axes[0, 0].set_title("Node Count Distribution", fontsize=14, fontweight='bold')
#     axes[0, 0].set_xlabel("Number of Nodes")
#     axes[0, 0].set_ylabel("Frequency")
#     axes[0, 0].legend()
#     axes[0, 0].grid(alpha=0.3)
    
#     # Edge count distribution
#     axes[0, 1].hist(num_edges, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
#     axes[0, 1].axvline(num_edges.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {num_edges.mean():.1f}')
#     axes[0, 1].set_title("Edge Count Distribution", fontsize=14, fontweight='bold')
#     axes[0, 1].set_xlabel("Number of Edges")
#     axes[0, 1].set_ylabel("Frequency")
#     axes[0, 1].legend()
#     axes[0, 1].grid(alpha=0.3)
    
#     # Density distribution
#     axes[0, 2].hist(densities, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
#     axes[0, 2].axvline(densities.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {densities.mean():.3f}')
#     axes[0, 2].set_title("Graph Density Distribution", fontsize=14, fontweight='bold')
#     axes[0, 2].set_xlabel("Density")
#     axes[0, 2].set_ylabel("Frequency")
#     axes[0, 2].legend()
#     axes[0, 2].grid(alpha=0.3)
    
#     # Root node distribution
#     axes[1, 0].hist(num_roots, bins=range(int(num_roots.min()), int(num_roots.max())+2), 
#                     color='gold', edgecolor='black', alpha=0.7)
#     axes[1, 0].axvline(num_roots.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {num_roots.mean():.1f}')
#     axes[1, 0].set_title("Root Nodes per Graph", fontsize=14, fontweight='bold')
#     axes[1, 0].set_xlabel("Number of Root Nodes")
#     axes[1, 0].set_ylabel("Frequency")
#     axes[1, 0].legend()
#     axes[1, 0].grid(alpha=0.3)
    
#     # Intermediate node distribution
#     axes[1, 1].hist(num_intermediates, bins=30, color='plum', edgecolor='black', alpha=0.7)
#     axes[1, 1].axvline(num_intermediates.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {num_intermediates.mean():.1f}')
#     axes[1, 1].set_title("Intermediate Nodes per Graph", fontsize=14, fontweight='bold')
#     axes[1, 1].set_xlabel("Number of Intermediate Nodes")
#     axes[1, 1].set_ylabel("Frequency")
#     axes[1, 1].legend()
#     axes[1, 1].grid(alpha=0.3)
    
#     # Leaf node distribution
#     axes[1, 2].hist(num_leaves, bins=30, color='salmon', edgecolor='black', alpha=0.7)
#     axes[1, 2].axvline(num_leaves.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {num_leaves.mean():.1f}')
#     axes[1, 2].set_title("Leaf Nodes per Graph", fontsize=14, fontweight='bold')
#     axes[1, 2].set_xlabel("Number of Leaf Nodes")
#     axes[1, 2].set_ylabel("Frequency")
#     axes[1, 2].legend()
#     axes[1, 2].grid(alpha=0.3)
    
#     plt.tight_layout()
#     if SAVE_PLOTS:
#         plt.savefig(os.path.join(OUTPUT_DIR, "structural_distributions.png"), dpi=300, bbox_inches='tight')
#     plt.show()
    
#     # Figure 2: Root Node Prior Probability Analysis
#     if all_root_priors:
#         fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
        
#         # Histogram
#         axes[0].hist(all_root_priors, bins=np.linspace(0, 1, 21), 
#                      color='mediumseagreen', edgecolor='black', alpha=0.7)
#         axes[0].axvline(np.mean(all_root_priors), color='red', linestyle='--', 
#                         linewidth=2, label=f'Mean: {np.mean(all_root_priors):.3f}')
#         axes[0].axvline(0.5, color='gray', linestyle=':', linewidth=2, label='0.5 (Uniform)')
#         axes[0].set_title("Root Node Prior Probability Distribution", fontsize=14, fontweight='bold')
#         axes[0].set_xlabel("Prior Probability P(State=1)")
#         axes[0].set_ylabel("Frequency")
#         axes[0].legend()
#         axes[0].grid(alpha=0.3)
        
#         # Box plot showing quartiles
#         axes[1].boxplot(all_root_priors, vert=True, patch_artist=True,
#                         boxprops=dict(facecolor='mediumseagreen', alpha=0.7),
#                         medianprops=dict(color='red', linewidth=2))
#         axes[1].set_ylabel("Prior Probability P(State=1)")
#         axes[1].set_title("Root Node Prior Probability Box Plot", fontsize=14, fontweight='bold')
#         axes[1].grid(alpha=0.3, axis='y')
#         axes[1].set_xticklabels(['All Root Nodes'])
        
#         plt.tight_layout()
#         if SAVE_PLOTS:
#             plt.savefig(os.path.join(OUTPUT_DIR, "root_prior_analysis.png"), dpi=300, bbox_inches='tight')
#         plt.show()
    
#     # Figure 3: Correlation Analysis
#     fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Nodes vs Edges scatter
#     axes[0].scatter(num_nodes, num_edges, alpha=0.3, s=10, color='steelblue')
#     axes[0].set_xlabel("Number of Nodes")
#     axes[0].set_ylabel("Number of Edges")
#     axes[0].set_title("Nodes vs Edges Relationship", fontsize=14, fontweight='bold')
#     axes[0].grid(alpha=0.3)
    
#     # Add correlation coefficient
#     corr = np.corrcoef(num_nodes, num_edges)[0, 1]
#     axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
#                  transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
#                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     # Node composition stacked bar
#     node_types = np.vstack([num_roots, num_intermediates, num_leaves])
#     x_pos = np.arange(min(10, len(results)))
    
#     axes[1].bar(x_pos, num_roots[:len(x_pos)], label='Roots', color='gold', alpha=0.8)
#     axes[1].bar(x_pos, num_intermediates[:len(x_pos)], bottom=num_roots[:len(x_pos)], 
#                 label='Intermediates', color='plum', alpha=0.8)
#     axes[1].bar(x_pos, num_leaves[:len(x_pos)], 
#                 bottom=num_roots[:len(x_pos)]+num_intermediates[:len(x_pos)], 
#                 label='Leaves', color='salmon', alpha=0.8)
#     axes[1].set_xlabel("Graph Sample Index")
#     axes[1].set_ylabel("Number of Nodes")
#     axes[1].set_title("Node Type Composition (First 10 Graphs)", fontsize=14, fontweight='bold')
#     axes[1].legend()
#     axes[1].grid(alpha=0.3, axis='y')
    
#     plt.tight_layout()
#     if SAVE_PLOTS:
#         plt.savefig(os.path.join(OUTPUT_DIR, "correlation_analysis.png"), dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print(f"\n✅ Analysis complete! Visualizations saved to '{OUTPUT_DIR}/' directory")

# if __name__ == "__main__":
#     main()
"""
Comprehensive Visualization Suite for Synthetic Bayesian Network Generation
===========================================================================
This script creates publication-quality visualizations showing:
1. Generation process flow diagram
2. Beta distribution analysis for root priors
3. Node/edge distributions
4. Sample graphs of different sizes
5. Structural properties analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
import seaborn as sns
import networkx as nx
from scipy import stats
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = "generation_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_generation_process_diagram():
    """Create a high-level UML-style diagram of the generation process"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Bayesian Network Generation Pipeline', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Define boxes with their positions and labels
    boxes = [
        # Stage 1: Structure Generation
        {'pos': (2, 8), 'width': 3, 'height': 0.8, 'label': '1. Structure Generation',
         'color': '#FFE5E5', 'details': ['• Sample nodes: [6, 200]', '• Depth: [3, 10]', '• Children: [1, 4]']},
        
        # Stage 2: Node Classification
        {'pos': (2, 6.5), 'width': 3, 'height': 0.8, 'label': '2. Node Classification',
         'color': '#E5F2FF', 'details': ['• Root nodes (no parents)', '• Intermediate nodes', '• Leaf nodes (no children)']},
        
        # Stage 3: CPD Assignment
        {'pos': (2, 5), 'width': 3, 'height': 0.8, 'label': '3. CPD Assignment',
         'color': '#E5FFE5', 'details': ['• 70% binary, 20% ternary', '• 10% higher cardinality']},
        
        # Stage 4: Root Priors
        {'pos': (6.5, 8), 'width': 3, 'height': 0.8, 'label': '4. Root Prior Generation',
         'color': '#FFF5E5', 'details': ['• Beta(0.1, 10): Rare events', '• Beta(5, 2): Common', '• Beta(2, 2): Moderate']},
        
        # Stage 5: Parent-Child CPDs
        {'pos': (6.5, 6.5), 'width': 3, 'height': 0.8, 'label': '5. Conditional CPDs',
         'color': '#F0E5FF', 'details': ['• Noisy-OR (25%)', '• Deterministic (15%)', '• Inhibitor (15%)', '• Mixed Strong (25%)', '• Weak (20%)']},
        
        # Final Output
        {'pos': (4.25, 3.5), 'width': 3.5, 'height': 0.8, 'label': '6. Final Bayesian Network',
         'color': '#E5E5E5', 'details': ['• Valid BN structure', '• Normalized CPDs', '• JSON export']}
    ]
    
    # Draw boxes
    for box in boxes:
        # Main box
        rect = FancyBboxPatch(
            box['pos'], box['width'], box['height'],
            boxstyle="round,pad=0.02",
            facecolor=box['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Label
        ax.text(box['pos'][0] + box['width']/2, box['pos'][1] + box['height']/2 + 0.1,
                box['label'], fontsize=11, fontweight='bold', ha='center')
        
        # Details
        for i, detail in enumerate(box['details']):
            ax.text(box['pos'][0] + box['width']/2, 
                    box['pos'][1] + box['height']/2 - 0.15 - i*0.12,
                    detail, fontsize=8, ha='center', style='italic')
    
    # Draw arrows
    arrows = [
        # Vertical flow on left
        ((3.5, 7.7), (3.5, 6.3)),
        ((3.5, 6.2), (3.5, 4.8)),
        
        # Horizontal connections
        ((5, 7.6), (6.5, 7.6)),
        ((5, 6.1), (6.5, 6.1)),
        
        # Converge to final
        ((3.5, 4.7), (4.25, 3.3)),
        ((8, 6.2), (7.75, 3.3))
    ]
    
    for start, end in arrows:
        arrow = FancyArrow(start[0], start[1], 
                          end[0]-start[0], end[1]-start[1],
                          width=0.03, head_width=0.1, head_length=0.08,
                          fc='gray', ec='gray')
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/generation_process_diagram.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved generation process diagram")


def visualize_beta_distributions():
    """Visualize the different beta distributions used for root node priors"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Beta Distributions for Root Node Prior Generation', fontsize=16, fontweight='bold')
    
    x = np.linspace(0, 1, 1000)
    
    # Different beta configurations used in generation
    configs = [
        {'params': (0.1, 10), 'label': 'Rare Events\nβ(0.1, 10)', 'color': 'red', 'use': '60% of roots'},
        {'params': (5, 2), 'label': 'Common Events\nβ(5, 2)', 'color': 'green', 'use': '20% of roots'},
        {'params': (2, 2), 'label': 'Moderate\nβ(2, 2)', 'color': 'blue', 'use': '20% of roots'},
        {'params': (0.5, 5), 'label': 'Alternative Rare\nβ(0.5, 5)', 'color': 'orange', 'use': 'Alternative'},
        {'params': (1, 1), 'label': 'Uniform\nβ(1, 1)', 'color': 'purple', 'use': 'Reference'},
        {'params': (2, 5), 'label': 'Slightly Rare\nβ(2, 5)', 'color': 'brown', 'use': 'Variation'}
    ]
    
    axes = axes.flatten()
    
    for idx, config in enumerate(configs):
        ax = axes[idx]
        alpha, beta = config['params']
        y = stats.beta.pdf(x, alpha, beta)
        
        ax.fill_between(x, y, alpha=0.3, color=config['color'])
        ax.plot(x, y, linewidth=2, color=config['color'])
        
        # Add statistics
        mean = alpha / (alpha + beta)
        mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else np.nan
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        
        ax.set_title(config['label'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')
        
        # Add text box with stats
        stats_text = f"Mean: {mean:.3f}\n"
        if not np.isnan(mode):
            stats_text += f"Mode: {mode:.3f}\n"
        stats_text += f"Var: {variance:.3f}\n{config['use']}"
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/beta_distributions_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved beta distributions analysis")


def analyze_generated_graphs(folder="generated_graphs"):
    """Analyze the actual generated graphs"""
    print("\n📊 Analyzing generated graphs...")
    
    # Load all graphs
    graph_files = list(Path(folder).glob("detailed_graph_*.json"))
    if not graph_files:
        print("⚠ No graph files found!")
        return None
    
    data = []
    root_priors_all = []
    
    for file in tqdm(graph_files[:40000], desc="Loading graphs"):  # Limit to 1000 for speed
        with open(file, 'r') as f:
            graph = json.load(f)
            
        num_nodes = len(graph['nodes'])
        num_edges = len(graph['edges'])
        
        # Count node types
        roots = len(graph['node_types']['roots'])
        leaves = len(graph['node_types']['leaves'])
        intermediates = len(graph['node_types']['intermediates'])
        
        # Extract root priors
        for node_id, node_info in graph['nodes'].items():
            if node_info['type'] == 'root':
                cpd = node_info.get('cpd', {})
                if 'values' in cpd:
                    values = cpd['values']
                    if len(values) >= 2:
                        # Assuming binary, get P(State=1)
                        root_priors_all.append(values[1])
        
        data.append({
            'nodes': num_nodes,
            'edges': num_edges,
            'roots': roots,
            'leaves': leaves,
            'intermediates': intermediates,
            'density': num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        })
    
    df = pd.DataFrame(data)
    return df, root_priors_all


def create_distribution_plots(df, root_priors):
    """Create comprehensive distribution plots"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Synthetic Bayesian Network Dataset Characteristics', fontsize=16, fontweight='bold')
    
    # 1. Node count distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['nodes'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df['nodes'].mean(), color='red', linestyle='--', label=f'Mean: {df["nodes"].mean():.1f}')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Node Count Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Edge count distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df['edges'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.axvline(df['edges'].mean(), color='red', linestyle='--', label=f'Mean: {df["edges"].mean():.1f}')
    ax2.set_xlabel('Number of Edges')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Edge Count Distribution')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Density distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df['density'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.axvline(df['density'].mean(), color='red', linestyle='--', label=f'Mean: {df["density"].mean():.4f}')
    ax3.set_xlabel('Graph Density')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Density Distribution')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Node types stacked
    ax4 = fig.add_subplot(gs[1, 0])
    node_types = df[['roots', 'intermediates', 'leaves']].mean()
    colors = ['gold', 'plum', 'salmon']
    ax4.bar(range(3), node_types, color=colors, edgecolor='black')
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(['Roots', 'Intermediates', 'Leaves'])
    ax4.set_ylabel('Average Count')
    ax4.set_title('Average Node Type Distribution')
    ax4.grid(alpha=0.3, axis='y')
    
    # 5. Root prior distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if root_priors:
        ax5.hist(root_priors, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
        ax5.axvline(np.mean(root_priors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(root_priors):.3f}')
        ax5.axvline(0.5, color='gray', linestyle=':', label='0.5 (uniform)')
        ax5.set_xlabel('Root Prior P(State=1)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Root Node Prior Distribution')
        ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Nodes vs Edges scatter
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(df['nodes'], df['edges'], 
                         c=df['density'], cmap='viridis', 
                         alpha=0.6, s=20)
    ax6.set_xlabel('Number of Nodes')
    ax6.set_ylabel('Number of Edges')
    ax6.set_title('Nodes vs Edges (colored by density)')
    plt.colorbar(scatter, ax=ax6, label='Density')
    ax6.grid(alpha=0.3)
    
    # 7. Node composition percentages
    ax7 = fig.add_subplot(gs[2, 0])
    total_nodes = df[['roots', 'intermediates', 'leaves']].sum(axis=1)
    root_pct = (df['roots'] / total_nodes * 100).mean()
    int_pct = (df['intermediates'] / total_nodes * 100).mean()
    leaf_pct = (df['leaves'] / total_nodes * 100).mean()
    
    wedges, texts, autotexts = ax7.pie([root_pct, int_pct, leaf_pct], 
                                        labels=['Roots', 'Intermediates', 'Leaves'],
                                        colors=['gold', 'plum', 'salmon'],
                                        autopct='%1.1f%%',
                                        startangle=90)
    ax7.set_title('Average Node Type Composition')
    
    # 8. Size categories
    ax8 = fig.add_subplot(gs[2, 1])
    size_cats = pd.cut(df['nodes'], bins=[0, 20, 50, 100, 200], 
                       labels=['Small\n(≤20)', 'Medium\n(21-50)', 'Large\n(51-100)', 'XLarge\n(>100)'])
    size_counts = size_cats.value_counts()
    ax8.bar(range(len(size_counts)), size_counts.values, 
           color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
    ax8.set_xticks(range(len(size_counts)))
    ax8.set_xticklabels(size_counts.index)
    ax8.set_ylabel('Count')
    ax8.set_title('Graph Size Distribution')
    ax8.grid(alpha=0.3, axis='y')
    
    # 9. Root prior ranges
    ax9 = fig.add_subplot(gs[2, 2])
    if root_priors:
        prior_ranges = pd.cut(root_priors, bins=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                             labels=['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'])
        range_counts = prior_ranges.value_counts().sort_index()
        ax9.bar(range(len(range_counts)), range_counts.values, 
               color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(range_counts))))
        ax9.set_xticks(range(len(range_counts)))
        ax9.set_xticklabels(range_counts.index, rotation=45)
        ax9.set_ylabel('Count')
        ax9.set_title('Root Prior Probability Ranges')
    ax9.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/distribution_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved distribution analysis")


def create_sample_graphs():
    """Create visualizations of sample graphs of different sizes"""
    print("\n🎨 Creating sample graph visualizations...")
    
    # Find graphs closest to target sizes
    target_sizes = [10, 25, 50, 100]
    folder = "generated_graphs"
    
    graph_files = list(Path(folder).glob("detailed_graph_*.json"))
    if not graph_files:
        print("⚠ No graphs found for samples")
        return
    
    # Find best matches for each size
    selected_graphs = []
    for target in target_sizes:
        best_match = None
        best_diff = float('inf')
        
        for file in graph_files:
            with open(file, 'r') as f:
                graph = json.load(f)
            num_nodes = len(graph['nodes'])
            diff = abs(num_nodes - target)
            
            if diff < best_diff:
                best_diff = diff
                best_match = (file, graph, num_nodes)
        
        if best_match:
            selected_graphs.append(best_match)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    fig.suptitle('Sample Bayesian Networks of Different Sizes', fontsize=16, fontweight='bold')
    
    for idx, (file, graph_data, size) in enumerate(selected_graphs):
        ax = axes[idx]
        
        # Create NetworkX graph
        G = nx.DiGraph()
        for edge_data in graph_data['edges']:
            G.add_edge(edge_data['source'], edge_data['target'])
        
        # Set node colors based on type
        node_colors = []
        for node in G.nodes():
            node_info = graph_data['nodes'][str(node)]
            if node_info['type'] == 'root':
                node_colors.append('gold')
            elif node_info['type'] == 'leaf':
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')
        
        # Layout
        if size <= 25:
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=max(10, 300/np.sqrt(size)), alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                              alpha=0.4, arrows=True, 
                              arrowsize=max(5, 20/np.sqrt(size)),
                              width=max(0.5, 2/np.sqrt(size)))
        
        # Stats
        roots = len(graph_data['node_types']['roots'])
        leaves = len(graph_data['node_types']['leaves']) 
        intermediates = len(graph_data['node_types']['intermediates'])
        
        ax.set_title(f'Network with {size} nodes\n' + 
                    f'R:{roots} / I:{intermediates} / L:{leaves}',
                    fontsize=12)
        ax.axis('off')
        
        # Add legend for first plot
        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gold', label='Root'),
                Patch(facecolor='lightblue', label='Intermediate'),
                Patch(facecolor='lightcoral', label='Leaf')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sample_graphs.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved sample graphs")


def create_cpd_pattern_visualization():
    """Visualize the different CPD patterns used"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CPD Generation Patterns for Parent-Child Relationships', 
                fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    patterns = [
        {
            'name': 'Noisy-OR (25%)',
            'description': 'Probability increases\nwith active parents',
            'color': 'red',
            'example': [0.1, 0.6, 0.6, 0.95]  # No parent, 1 parent, other parent, both
        },
        {
            'name': 'Deterministic (15%)',
            'description': 'Strong mapping\nwith minimal noise',
            'color': 'blue', 
            'example': [0.02, 0.98, 0.98, 0.02]
        },
        {
            'name': 'Inhibitor (15%)',
            'description': 'One parent can\nblock effect',
            'color': 'orange',
            'example': [0.1, 0.1, 0.9, 0.1]
        },
        {
            'name': 'Mixed Strong (25%)',
            'description': 'Weighted parent\ninfluence',
            'color': 'green',
            'example': [0.2, 0.7, 0.6, 0.9]
        },
        {
            'name': 'Weak (20%)',
            'description': 'Minimal parent\ndependence',
            'color': 'purple',
            'example': [0.4, 0.5, 0.45, 0.55]
        }
    ]
    
    parent_configs = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
    
    for idx, pattern in enumerate(patterns):
        ax = axes[idx]
        
        # Bar plot showing probabilities for different parent configurations
        bars = ax.bar(range(4), pattern['example'], color=pattern['color'], alpha=0.7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(parent_configs)
        ax.set_ylim(0, 1)
        ax.set_ylabel('P(Child=1|Parents)')
        ax.set_xlabel('Parent Configuration')
        ax.set_title(pattern['name'], fontweight='bold')
        ax.text(0.5, 0.85, pattern['description'], 
               transform=ax.transAxes, ha='center',
               fontsize=10, style='italic')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, pattern['example']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Remove last empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cpd_patterns.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved CPD patterns visualization")


def create_summary_statistics_table(df, root_priors):
    """Create a formatted summary statistics table"""
    stats = {
        'Metric': [],
        'Mean': [],
        'Std Dev': [],
        'Min': [],
        'Q1': [],
        'Median': [],
        'Q3': [],
        'Max': []
    }
    
    # Add metrics
    metrics = [
        ('Nodes', df['nodes']),
        ('Edges', df['edges']),
        ('Density', df['density']),
        ('Root Nodes', df['roots']),
        ('Intermediate Nodes', df['intermediates']),
        ('Leaf Nodes', df['leaves']),
        ('Root Priors', root_priors if root_priors else [])
    ]
    
    for name, data in metrics:
        if len(data) > 0:
            stats['Metric'].append(name)
            stats['Mean'].append(f'{np.mean(data):.2f}')
            stats['Std Dev'].append(f'{np.std(data):.2f}')
            stats['Min'].append(f'{np.min(data):.2f}')
            stats['Q1'].append(f'{np.percentile(data, 25):.2f}')
            stats['Median'].append(f'{np.median(data):.2f}')
            stats['Q3'].append(f'{np.percentile(data, 75):.2f}')
            stats['Max'].append(f'{np.max(data):.2f}')
    
    stats_df = pd.DataFrame(stats)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=stats_df.values,
                    colLabels=stats_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15] + [0.12]*7)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_df) + 1):
        for j in range(len(stats_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f1f1f2')
    
    plt.title('Summary Statistics of Generated Bayesian Networks', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(f"{OUTPUT_DIR}/summary_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved summary statistics table")


def main():
    print("="*60)
    print("SYNTHETIC BAYESIAN NETWORK GENERATION VISUALIZATION")
    print("="*60)
    
    # 1. Create generation process diagram
    create_generation_process_diagram()
    
    # 2. Visualize beta distributions
    visualize_beta_distributions()
    
    # 3. Analyze actual generated graphs
    result = analyze_generated_graphs()
    if result:
        df, root_priors = result
        
        # 4. Create distribution plots
        create_distribution_plots(df, root_priors)
        
        # 5. Create sample graphs
        create_sample_graphs()
        
        # 6. Create CPD pattern visualization
        create_cpd_pattern_visualization()
        
        # 7. Create summary statistics table
        create_summary_statistics_table(df, root_priors)
        
        print(f"\n✅ All visualizations saved to '{OUTPUT_DIR}/'")
        print("\nGenerated files:")
        for file in os.listdir(OUTPUT_DIR):
            print(f"  • {file}")
    else:
        print("⚠ Could not analyze graphs - generating example visualizations only")
        create_cpd_pattern_visualization()


if __name__ == "__main__":
    main()