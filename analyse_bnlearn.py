"""
Visualize the relationship between network size and prediction error
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("benchmark_results/results.json") as f:
    results = json.load(f)

# Extract data
networks = results["per_network_results"]

sizes = [n["num_nodes"] for n in networks]
errors = [n["absolute_error"] for n in networks]
names = [n["network_name"] for n in networks]
gt_probs = [n["ground_truth_prob"] for n in networks]
pred_probs = [n["prediction_prob"] for n in networks]

# Create figure with 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Error vs Network Size
ax1 = axes[0, 0]
scatter = ax1.scatter(sizes, errors, c=errors, cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black')
ax1.set_xlabel('Network Size (nodes)', fontsize=12)
ax1.set_ylabel('Absolute Error', fontsize=12)
ax1.set_title('Prediction Error vs Network Size', fontsize=14, fontweight='bold')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)

# Add threshold line
ax1.axhline(y=0.1, color='green', linestyle='--', linewidth=2, label='Good (<0.1)')
ax1.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, label='Acceptable (<0.3)')
ax1.legend()

# Annotate outliers
for i, (s, e, name) in enumerate(zip(sizes, errors, names)):
    if s > 500 or e > 0.8:
        ax1.annotate(name, (s, e), fontsize=8, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')

plt.colorbar(scatter, ax=ax1, label='Error')

# 2. Prediction Distribution by Size Category
ax2 = axes[0, 1]

small = [p for s, p in zip(sizes, pred_probs) if s < 50]
medium = [p for s, p in zip(sizes, pred_probs) if 50 <= s < 200]
large = [p for s, p in zip(sizes, pred_probs) if s >= 200]

bp = ax2.boxplot([small, medium, large], labels=['<50', '50-200', '200+'],
                  patch_artist=True, showmeans=True)

for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

ax2.set_xlabel('Network Size Category', fontsize=12)
ax2.set_ylabel('Predicted Probability', fontsize=12)
ax2.set_title('Prediction Distribution by Network Size', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Expected mean ~0.5')
ax2.legend()

# 3. Ground Truth vs Prediction colored by size
ax3 = axes[1, 0]

# Color by size
size_colors = np.array(sizes)
scatter = ax3.scatter(gt_probs, pred_probs, c=size_colors, s=100, alpha=0.6,
                     cmap='viridis', edgecolors='black', norm=plt.matplotlib.colors.LogNorm())

ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
ax3.set_xlabel('Ground Truth Probability', fontsize=12)
ax3.set_ylabel('Predicted Probability', fontsize=12)
ax3.set_title('Predictions (colored by network size)', fontsize=14, fontweight='bold')
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.grid(True, alpha=0.3)
ax3.legend()

cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Network Size (nodes)', fontsize=10)

# 4. Statistics by size
ax4 = axes[1, 1]

# Calculate stats by size category
categories = ['<50', '50-200', '200+', 'All']
cat_masks = [
    np.array(sizes) < 50,
    (np.array(sizes) >= 50) & (np.array(sizes) < 200),
    np.array(sizes) >= 200,
    np.ones(len(sizes), dtype=bool)
]

maes = []
within_10pcts = []

for mask in cat_masks:
    if mask.sum() == 0:
        maes.append(0)
        within_10pcts.append(0)
        continue
    
    cat_errors = np.array(errors)[mask]
    maes.append(np.mean(cat_errors))
    within_10pcts.append(np.mean(cat_errors <= 0.1) * 100)

x_pos = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, maes, width, label='MAE', color='steelblue', alpha=0.7)
bars2 = ax4.bar(x_pos + width/2, [w/100 for w in within_10pcts], width, 
                label='Within 10%', color='orange', alpha=0.7)

ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Performance Metrics by Network Size', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(categories)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('benchmark_results/size_effect_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved to benchmark_results/size_effect_analysis.png")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for cat, mask in zip(categories, cat_masks):
    if mask.sum() == 0:
        continue
    
    cat_sizes = np.array(sizes)[mask]
    cat_errors = np.array(errors)[mask]
    cat_gt = np.array(gt_probs)[mask]
    cat_pred = np.array(pred_probs)[mask]
    
    print(f"\n{cat} nodes ({mask.sum()} networks):")
    print(f"  Size range:        {cat_sizes.min():.0f} - {cat_sizes.max():.0f}")
    print(f"  MAE:               {np.mean(cat_errors):.4f}")
    print(f"  Median Error:      {np.median(cat_errors):.4f}")
    print(f"  Max Error:         {np.max(cat_errors):.4f}")
    print(f"  Within 5%:         {np.mean(cat_errors <= 0.05)*100:.1f}%")
    print(f"  Within 10%:        {np.mean(cat_errors <= 0.10)*100:.1f}%")
    print(f"  Mean GT Prob:      {np.mean(cat_gt):.4f}")
    print(f"  Mean Pred Prob:    {np.mean(cat_pred):.4f}")
    print(f"  Pred Std:          {np.std(cat_pred):.4f}")

print("\n" + "="*70)
print("KEY FINDING:")
print("="*70)
print("Performance degrades dramatically as network size increases!")
print(f"Small networks (<50):  MAE = {np.mean(np.array(errors)[cat_masks[0]]):.4f}")
print(f"Large networks (200+): MAE = {np.mean(np.array(errors)[cat_masks[2]]):.4f}")
print(f"Degradation: {np.mean(np.array(errors)[cat_masks[2]]) / np.mean(np.array(errors)[cat_masks[0]]):.1f}x worse!")

# Find best and worst
best_idx = np.argmin(errors)
worst_idx = np.argmax(errors)

print(f"\n✅ Best:  {names[best_idx]} ({sizes[best_idx]} nodes) - Error: {errors[best_idx]:.4f}")
print(f"❌ Worst: {names[worst_idx]} ({sizes[worst_idx]} nodes) - Error: {errors[worst_idx]:.4f}")

plt.show()