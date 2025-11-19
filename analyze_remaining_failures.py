"""
Identify which graphs are still failing after fine-tuning
"""

import json
import numpy as np

# Load results
with open('benchmark_results/results.json', 'r') as f:
    results = json.load(f)

per_network = results['per_network_results']

# Convert to probability space
failures = []
for net in per_network:
    truth_prob = net['ground_truth_prob']
    pred_prob = net['prediction_prob']
    error = abs(truth_prob - pred_prob)
    
    failures.append({
        'name': net['network_name'],
        'truth': truth_prob,
        'pred': pred_prob,
        'error': error,
        'nodes': net['num_nodes'],
        'relative_error': error / (truth_prob + 1e-8)
    })

# Sort by error
failures.sort(key=lambda x: x['error'], reverse=True)

print("="*70)
print("TOP 10 REMAINING FAILURES")
print("="*70)

for i, f in enumerate(failures[:10], 1):
    print(f"\n{i}. {f['name']} ({f['nodes']} nodes)")
    print(f"   Truth: {f['truth']:.4f}, Pred: {f['pred']:.4f}")
    print(f"   Error: {f['error']:.4f} ({f['relative_error']:.1f}x)")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

high_errors = [f for f in failures if f['error'] > 0.3]
print(f"Graphs with error > 0.3: {len(high_errors)}/24")

extreme_truth = [f for f in failures if f['truth'] < 0.1 or f['truth'] > 0.9]
print(f"Graphs with extreme truth values: {len(extreme_truth)}/24")

print("\nPatterns:")
for f in high_errors:
    if f['truth'] < 0.1:
        print(f"  • {f['name']}: Very low truth ({f['truth']:.4f})")
    elif f['truth'] > 0.9:
        print(f"  • {f['name']}: Very high truth ({f['truth']:.4f})")