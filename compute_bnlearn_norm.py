"""
Compute normalization statistics specifically for BNLearn graphs
Run this ONCE, then use the generated stats in benchmarking
"""

import torch
import numpy as np
import glob
import os
from pathlib import Path
from BIF_data_debugging import BenchmarkDatasetProcessor

def compute_bnlearn_normalization_stats(
    bif_directory="dataset_bif_files",
    output_path="bnlearn_norm_stats.pt",
    config_path="config.yaml"
):
    """
    Extract features from ALL BNLearn graphs and compute normalization stats
    """
    print("="*70)
    print("COMPUTING BNLEARN NORMALIZATION STATISTICS")
    print("="*70)
    
    # Initialize processor WITHOUT normalization
    processor = BenchmarkDatasetProcessor(config_path, verbose=True)
    processor.norm_stats = None  # Disable normalization during extraction
    
    bif_files = glob.glob(os.path.join(bif_directory, "*.bif"))
    print(f"\nFound {len(bif_files)} BNLearn graphs")
    
    all_features = []
    
    print("\nExtracting features from BNLearn graphs...")
    for i, bif_path in enumerate(bif_files, 1):
        network_name = Path(bif_path).stem
        print(f"  [{i}/{len(bif_files)}] {network_name}...", end=" ")
        
        try:
            # Process WITHOUT cache to get raw features
            graph, meta = processor.process_bif_to_graph(
                bif_path, network_name, use_cache=False
            )
            
            if graph is not None:
                all_features.append(graph.x)  # Collect features
                print(f"✓ ({meta['num_nodes']} nodes)")
            else:
                print("⊘ Filtered")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if len(all_features) == 0:
        raise RuntimeError("No BNLearn graphs processed!")
    
    # Stack all features
    all_features_tensor = torch.cat(all_features, dim=0)
    print(f"\n✓ Collected {all_features_tensor.shape[0]} nodes from {len(all_features)} graphs")
    print(f"  Feature shape: {all_features_tensor.shape}")
    
    # ===== COMPUTE STATISTICS (SAME STRUCTURE AS TRAINING) =====
    
    # Base features: indices 1-8 (skip node_type at index 0)
    base_indices = list(range(1, 9))
    base_features = all_features_tensor[:, base_indices]
    
    base_stats = {
        'indices': base_indices,
        'mean': base_features.mean(dim=0).numpy().tolist(),
        'std': base_features.std(dim=0).numpy().tolist()
    }
    
    # CPD features: indices 10-19 (skip argmax at index 15, which is 10+5)
    cpd_features = all_features_tensor[:, 10:20]
    cpd_to_normalize = torch.cat([
        cpd_features[:, :5],   # indices 10-14
        cpd_features[:, 6:]    # indices 16-19 (skip index 15 which is argmax)
    ], dim=1)
    
    cpd_stats = {
        'mean': cpd_to_normalize.mean(dim=0).numpy().tolist(),
        'std': cpd_to_normalize.std(dim=0).numpy().tolist()
    }
    
    # Distance feature: index 24 (last feature)
    distance_features = all_features_tensor[:, 24]
    
    distance_stats = {
        'mean': distance_features.mean().item(),
        'std': distance_features.std().item()
    }
    
    # Package stats
    bnlearn_stats = {
        'base': base_stats,
        'cpd': cpd_stats,
        'distance': distance_stats,
        'source': 'bnlearn_graphs',
        'num_graphs': len(all_features),
        'num_nodes_total': all_features_tensor.shape[0]
    }
    
    # Save
    torch.save(bnlearn_stats, output_path)
    
    print("\n" + "="*70)
    print(f"✓ SAVED BNLEARN NORMALIZATION STATS")
    print(f"  Output: {output_path}")
    print(f"  Graphs: {len(all_features)}")
    print(f"  Total nodes: {all_features_tensor.shape[0]}")
    print("="*70)
    
    # Print summary
    print("\nStatistics Summary:")
    print(f"  Base features mean range: [{min(base_stats['mean']):.4f}, {max(base_stats['mean']):.4f}]")
    print(f"  Base features std range:  [{min(base_stats['std']):.4f}, {max(base_stats['std']):.4f}]")
    print(f"  CPD mean range: [{min(cpd_stats['mean']):.4f}, {max(cpd_stats['mean']):.4f}]")
    print(f"  CPD std range:  [{min(cpd_stats['std']):.4f}, {max(cpd_stats['std']):.4f}]")
    print(f"  Distance mean: {distance_stats['mean']:.4f}")
    print(f"  Distance std:  {distance_stats['std']:.4f}")
    
    return bnlearn_stats


if __name__ == "__main__":
    stats = compute_bnlearn_normalization_stats()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Copy bnlearn_norm_stats.pt to your benchmark directory")
    print("2. In benchmark_script.py, change the normalization path:")
    print("   FROM: 'datasets/folds/fold_0_norm_stats.pt'")
    print("   TO:   'bnlearn_norm_stats.pt'")
    print("="*70)