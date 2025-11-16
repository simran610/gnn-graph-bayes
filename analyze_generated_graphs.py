import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import seaborn as sns

# ========= Configuration =========
FOLDER = "generated_graphs"
NUM_SAMPLE_VIS = 20  # Number of sample graphs to visualize
SAVE_PLOTS = True
OUTPUT_DIR = "analysis_results"
# =================================

def load_and_analyze_graph(file_path):
    """Efficiently loads and analyzes a single graph in one pass."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        nodes = data.get("nodes", {})
        edges = data.get("edges", [])
        
        num_nodes = len(nodes)
        num_edges = len(edges)
        
        # Count node types and collect root priors in one pass
        roots = 0
        leaves = 0
        intermediates = 0
        root_priors = []
        
        for node_id, node in nodes.items():
            node_type = node.get("type", "unknown")
            
            if node_type == "root":
                roots += 1
                # Extract root node prior probability
                cpd = node.get("cpd", {})
                if "values" in cpd:
                    try:
                        values = cpd["values"]
                        variable_card = cpd.get("variable_card", 2)
                        
                        # For root nodes, values is a flat list: [P(State=0), P(State=1), ...]
                        # We want P(State=1) which is at index 1
                        if isinstance(values, list) and len(values) >= variable_card and variable_card >= 2:
                            prior_prob = values[1]  # P(State=1)
                            root_priors.append(float(prior_prob))
                    except (IndexError, KeyError, TypeError, ValueError) as e:
                        pass
            elif node_type == "leaf":
                leaves += 1
            elif node_type == "intermediate":
                intermediates += 1
        
        # Calculate density
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        return {
            "nodes": num_nodes,
            "edges": num_edges,
            "density": density,
            "roots": roots,
            "leaves": leaves,
            "intermediates": intermediates,
            "root_priors": root_priors,
            "file": os.path.basename(file_path)
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    """Main analysis pipeline."""
    
    # Create output directory
    if SAVE_PLOTS and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 1. Find all graph files
    print("🔍 Scanning for graph files...")
    all_files = [
        os.path.join(FOLDER, f)
        for f in os.listdir(FOLDER)
        if f.startswith("detailed_graph_") and f.endswith(".json")
    ]
    all_files.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
    
    if not all_files:
        print(f"❌ No graph files found in '{FOLDER}'")
        return
    
    print(f"📂 Found {len(all_files)} graph files\n")
    
    # 2. Analyze all graphs efficiently
    print("📊 Analyzing graphs...")
    results = []
    for fpath in tqdm(all_files, desc="Processing"):
        result = load_and_analyze_graph(fpath)
        if result:
            results.append(result)
    
    # 3. Extract arrays for analysis
    num_nodes = np.array([r["nodes"] for r in results])
    num_edges = np.array([r["edges"] for r in results])
    densities = np.array([r["density"] for r in results])
    num_roots = np.array([r["roots"] for r in results])
    num_leaves = np.array([r["leaves"] for r in results])
    num_intermediates = np.array([r["intermediates"] for r in results])
    
    # Flatten all root priors
    all_root_priors = [p for r in results for p in r["root_priors"]]
    
    # 4. Print Summary Statistics
    print("\n" + "="*60)
    print("📈 DATASET SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Graphs Analyzed: {len(results)}")
    print(f"\n--- STRUCTURAL PROPERTIES ---")
    print(f"Nodes per Graph:")
    print(f"  Mean: {num_nodes.mean():.2f} | Median: {np.median(num_nodes):.0f}")
    print(f"  Min: {num_nodes.min()} | Max: {num_nodes.max()}")
    print(f"  Std Dev: {num_nodes.std():.2f}")
    
    print(f"\nEdges per Graph:")
    print(f"  Mean: {num_edges.mean():.2f} | Median: {np.median(num_edges):.0f}")
    print(f"  Min: {num_edges.min()} | Max: {num_edges.max()}")
    print(f"  Std Dev: {num_edges.std():.2f}")
    
    print(f"\nGraph Density:")
    print(f"  Mean: {densities.mean():.4f} | Median: {np.median(densities):.4f}")
    print(f"  Min: {densities.min():.4f} | Max: {densities.max():.4f}")
    
    print(f"\n--- NODE TYPE DISTRIBUTION ---")
    print(f"Root Nodes per Graph:")
    print(f"  Mean: {num_roots.mean():.2f} | Median: {np.median(num_roots):.0f}")
    print(f"  Min: {num_roots.min()} | Max: {num_roots.max()}")
    
    print(f"\nIntermediate Nodes per Graph:")
    print(f"  Mean: {num_intermediates.mean():.2f} | Median: {np.median(num_intermediates):.0f}")
    print(f"  Min: {num_intermediates.min()} | Max: {num_intermediates.max()}")
    
    print(f"\nLeaf Nodes per Graph:")
    print(f"  Mean: {num_leaves.mean():.2f} | Median: {np.median(num_leaves):.0f}")
    print(f"  Min: {num_leaves.min()} | Max: {num_leaves.max()}")
    
    print(f"\n--- ROOT NODE PRIOR PROBABILITIES ---")
    print(f"Total Root Nodes Analyzed: {len(all_root_priors)}")
    if all_root_priors:
        print(f"Prior P(State=1):")
        print(f"  Mean: {np.mean(all_root_priors):.4f} | Median: {np.median(all_root_priors):.4f}")
        print(f"  Min: {np.min(all_root_priors):.4f} | Max: {np.max(all_root_priors):.4f}")
        print(f"  Std Dev: {np.std(all_root_priors):.4f}")
        
        # Analysis of low probability roots
        low_prob_count = sum(1 for p in all_root_priors if p < 0.3)
        mid_prob_count = sum(1 for p in all_root_priors if 0.3 <= p <= 0.7)
        high_prob_count = sum(1 for p in all_root_priors if p > 0.7)
        
        print(f"\n  Low Probability (<0.3): {low_prob_count} ({100*low_prob_count/len(all_root_priors):.1f}%)")
        print(f"  Mid Probability (0.3-0.7): {mid_prob_count} ({100*mid_prob_count/len(all_root_priors):.1f}%)")
        print(f"  High Probability (>0.7): {high_prob_count} ({100*high_prob_count/len(all_root_priors):.1f}%)")
    
    print("="*60 + "\n")
    
    # 5. Create Comprehensive Visualizations
    print("🎨 Creating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Figure 1: Main Distributions (2x3 grid)
    fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Node count distribution
    axes[0, 0].hist(num_nodes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(num_nodes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {num_nodes.mean():.1f}')
    axes[0, 0].set_title("Node Count Distribution", fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel("Number of Nodes")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Edge count distribution
    axes[0, 1].hist(num_edges, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(num_edges.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {num_edges.mean():.1f}')
    axes[0, 1].set_title("Edge Count Distribution", fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel("Number of Edges")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Density distribution
    axes[0, 2].hist(densities, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(densities.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {densities.mean():.3f}')
    axes[0, 2].set_title("Graph Density Distribution", fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel("Density")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Root node distribution
    axes[1, 0].hist(num_roots, bins=range(int(num_roots.min()), int(num_roots.max())+2), 
                    color='gold', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(num_roots.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {num_roots.mean():.1f}')
    axes[1, 0].set_title("Root Nodes per Graph", fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel("Number of Root Nodes")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Intermediate node distribution
    axes[1, 1].hist(num_intermediates, bins=30, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(num_intermediates.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {num_intermediates.mean():.1f}')
    axes[1, 1].set_title("Intermediate Nodes per Graph", fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel("Number of Intermediate Nodes")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Leaf node distribution
    axes[1, 2].hist(num_leaves, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    axes[1, 2].axvline(num_leaves.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {num_leaves.mean():.1f}')
    axes[1, 2].set_title("Leaf Nodes per Graph", fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel("Number of Leaf Nodes")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(os.path.join(OUTPUT_DIR, "structural_distributions.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Root Node Prior Probability Analysis
    if all_root_priors:
        fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        axes[0].hist(all_root_priors, bins=np.linspace(0, 1, 21), 
                     color='mediumseagreen', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(all_root_priors), color='red', linestyle='--', 
                        linewidth=2, label=f'Mean: {np.mean(all_root_priors):.3f}')
        axes[0].axvline(0.5, color='gray', linestyle=':', linewidth=2, label='0.5 (Uniform)')
        axes[0].set_title("Root Node Prior Probability Distribution", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Prior Probability P(State=1)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot showing quartiles
        axes[1].boxplot(all_root_priors, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='mediumseagreen', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel("Prior Probability P(State=1)")
        axes[1].set_title("Root Node Prior Probability Box Plot", fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        axes[1].set_xticklabels(['All Root Nodes'])
        
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(os.path.join(OUTPUT_DIR, "root_prior_analysis.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Figure 3: Correlation Analysis
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Nodes vs Edges scatter
    axes[0].scatter(num_nodes, num_edges, alpha=0.3, s=10, color='steelblue')
    axes[0].set_xlabel("Number of Nodes")
    axes[0].set_ylabel("Number of Edges")
    axes[0].set_title("Nodes vs Edges Relationship", fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(num_nodes, num_edges)[0, 1]
    axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                 transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Node composition stacked bar
    node_types = np.vstack([num_roots, num_intermediates, num_leaves])
    x_pos = np.arange(min(10, len(results)))
    
    axes[1].bar(x_pos, num_roots[:len(x_pos)], label='Roots', color='gold', alpha=0.8)
    axes[1].bar(x_pos, num_intermediates[:len(x_pos)], bottom=num_roots[:len(x_pos)], 
                label='Intermediates', color='plum', alpha=0.8)
    axes[1].bar(x_pos, num_leaves[:len(x_pos)], 
                bottom=num_roots[:len(x_pos)]+num_intermediates[:len(x_pos)], 
                label='Leaves', color='salmon', alpha=0.8)
    axes[1].set_xlabel("Graph Sample Index")
    axes[1].set_ylabel("Number of Nodes")
    axes[1].set_title("Node Type Composition (First 10 Graphs)", fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(os.path.join(OUTPUT_DIR, "correlation_analysis.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Analysis complete! Visualizations saved to '{OUTPUT_DIR}/' directory")

if __name__ == "__main__":
    main()