"""
Compare feature distributions between training data and BNLearn
Helps understand domain shift
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from BIF_data_debugging import BenchmarkDatasetProcessor
import glob
from pathlib import Path

def compare_distributions():
    print("Comparing Training vs BNLearn Distributions...")
    
    # Load training stats
    train_stats = torch.load('datasets/folds/fold_0_norm_stats.pt', weights_only=False)
    
    # Load BNLearn stats
    try:
        bnlearn_stats = torch.load('bnlearn_norm_stats.pt', weights_only=False)
    except:
        print("❌ BNLearn stats not found! Run compute_bnlearn_norm.py first")
        return
    
    # Compare base features
    train_base_mean = np.array(train_stats['base']['mean'])
    train_base_std = np.array(train_stats['base']['std'])
    
    bnlearn_base_mean = np.array(bnlearn_stats['base']['mean'])
    bnlearn_base_std = np.array(bnlearn_stats['base']['std'])
    
    # Compute divergence
    mean_divergence = np.abs(train_base_mean - bnlearn_base_mean)
    std_ratio = bnlearn_base_std / (train_base_std + 1e-8)
    
    feature_names = [
        'in_degree', 'out_degree', 'betweenness', 
        'closeness', 'pagerank', 'degree_cent',
        'variable_card', 'num_parents'
    ]
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Mean divergence
    axes[0].bar(range(len(feature_names)), mean_divergence)
    axes[0].set_xticks(range(len(feature_names)))
    axes[0].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[0].set_ylabel('|Training Mean - BNLearn Mean|')
    axes[0].set_title('Feature Mean Divergence')
    axes[0].axhline(y=1.0, color='r', linestyle='--', label='Threshold (1.0)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Std ratio
    axes[1].bar(range(len(feature_names)), std_ratio)
    axes[1].set_xticks(range(len(feature_names)))
    axes[1].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[1].set_ylabel('BNLearn Std / Training Std')
    axes[1].set_title('Feature Std Ratio')
    axes[1].axhline(y=1.0, color='g', linestyle='--', label='Same std')
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[1].axhline(y=2.0, color='r', linestyle='--', alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=300)
    print("✓ Saved distribution_comparison.png")
    
    # Print summary
    print("\n" + "="*70)
    print("DISTRIBUTION COMPARISON SUMMARY")
    print("="*70)
    print(f"Features with large mean divergence (>1.0):")
    for i, name in enumerate(feature_names):
        if mean_divergence[i] > 1.0:
            print(f"  • {name}: {mean_divergence[i]:.3f}")
    
    print(f"\nFeatures with extreme std ratio (<0.5 or >2.0):")
    for i, name in enumerate(feature_names):
        if std_ratio[i] < 0.5 or std_ratio[i] > 2.0:
            print(f"  • {name}: {std_ratio[i]:.3f}x")
    
    print("="*70)

if __name__ == "__main__":
    compare_distributions()