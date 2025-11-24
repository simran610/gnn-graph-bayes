"""
GNN Diagnostic Script
=====================
This script helps diagnose why your GNN might be performing poorly.

Run this BEFORE the full benchmark to identify issues.
"""

import torch
import yaml
import numpy as np
from pathlib import Path
from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination
import sys

print("="*80)
print("GNN DIAGNOSTIC TOOL")
print("="*80)

# ===== CHECK 1: Required Files =====
print("\n[CHECK 1] Verifying required files...")
required_files = {
    'model': 'model_finetuned_bnlearn.pt',
    'norm': 'bnlearn_norm_stats.pt',
    'config': 'config.yaml',
    'model_code': 'graphsage_model.py'
}

missing = []
for name, path in required_files.items():
    if Path(path).exists():
        print(f"  ✓ {name}: {path}")
    else:
        print(f"  ✗ {name}: {path} NOT FOUND")
        missing.append(path)

if missing:
    print(f"\n❌ CRITICAL: Missing files: {missing}")
    print("   Cannot proceed without these files!")
    sys.exit(1)

# ===== CHECK 2: Model Architecture =====
print("\n[CHECK 2] Checking model architecture...")
try:
    model_state = torch.load('model_finetuned_bnlearn.pt', map_location='cpu', weights_only=False)
    
    # Try to infer input features from first layer
    if 'conv1.lin_l.weight' in model_state:
        first_layer_shape = model_state['conv1.lin_l.weight'].shape
        expected_features = first_layer_shape[1]
        print(f"  ✓ Model expects {expected_features} input features")
        
        if expected_features != 25:
            print(f"  ⚠ WARNING: Model expects {expected_features} features, but benchmark uses 25!")
            print(f"    This WILL cause incorrect predictions!")
    else:
        print("  ⚠ Could not determine expected features from model state")
    
    # Check output
    if 'out_lin.weight' in model_state:
        out_shape = model_state['out_lin.weight'].shape
        print(f"  ✓ Model output channels: {out_shape[0]}")
        if out_shape[0] != 1:
            print(f"  ⚠ WARNING: Output is not 1 channel (expected for root probability)")
    
except Exception as e:
    print(f"  ✗ Failed to load model: {e}")
    sys.exit(1)

# ===== CHECK 3: Config Settings =====
print("\n[CHECK 3] Checking config.yaml settings...")
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    mode = config.get('mode', 'unknown')
    use_log_prob = config.get('use_log_prob', False)
    
    print(f"  ✓ Mode: {mode}")
    print(f"  ✓ Log-probability: {use_log_prob}")
    
    if mode != 'root_probability':
        print(f"  ⚠ WARNING: Mode is '{mode}', benchmark expects 'root_probability'")
    
except Exception as e:
    print(f"  ✗ Failed to read config: {e}")

# ===== CHECK 4: Normalization Stats =====
print("\n[CHECK 4] Checking normalization stats...")
try:
    norm_stats = torch.load('bnlearn_norm_stats.pt', weights_only=False)
    
    required_keys = ['base', 'cpd', 'distance']
    for key in required_keys:
        if key in norm_stats:
            print(f"  ✓ {key}: present")
        else:
            print(f"  ✗ {key}: MISSING")
    
    # Check if stats make sense
    if 'base' in norm_stats:
        indices = norm_stats['base'].get('indices', [])
        mean = norm_stats['base'].get('mean', [])
        std = norm_stats['base'].get('std', [])
        print(f"    - Base features normalized: indices {indices}")
        print(f"    - Mean range: [{min(mean):.4f}, {max(mean):.4f}]")
        print(f"    - Std range: [{min(std):.4f}, {max(std):.4f}]")
        
        if any(s == 0 for s in std):
            print(f"  ⚠ WARNING: Some std values are 0! This will cause NaN!")
    
except Exception as e:
    print(f"  ✗ Failed to load normalization stats: {e}")
    print(f"  ⚠ WITHOUT normalization, GNN predictions will be WRONG!")

# ===== CHECK 5: Test on Simple Network =====
print("\n[CHECK 5] Testing on simple network (asia.bif)...")

# Find a simple test network
test_bif = None
for candidate in ['dataset_bif_files/asia.bif', 'asia.bif', 'dataset_bif_files/survey.bif']:
    if Path(candidate).exists():
        test_bif = candidate
        break

if not test_bif:
    print("  ⚠ No test network found, skipping test")
else:
    try:
        print(f"  Testing on: {test_bif}")
        
        # Load model
        from graphsage_model import GraphSAGE
        
        gnn = GraphSAGE(
            in_channels=expected_features,
            hidden_channels=128,
            out_channels=1,
            dropout=0.1,
            mode='root_probability',
            use_log_prob=use_log_prob
        )
        gnn.load_state_dict(model_state)
        gnn.eval()
        
        # Load network
        reader = BIFReader(test_bif)
        bn_model = reader.get_model()
        
        print(f"  Network: {bn_model.number_of_nodes()} nodes, {bn_model.number_of_edges()} edges")
        
        # Get root
        roots = []
        for node in bn_model.nodes():
            if len(list(bn_model.predecessors(node))) == 0 and len(list(bn_model.successors(node))) > 0:
                roots.append(node)
        
        if not roots:
            print("  ⚠ No root node found")
        else:
            root = roots[0]
            print(f"  Query node: {root}")
            
            # Ground truth
            inference = VariableElimination(bn_model)
            result = inference.query(variables=[root], show_progress=False)
            gt_prob = float(result.values[0])
            
            if use_log_prob:
                gt_value = np.log(max(gt_prob, 1e-10))
            else:
                gt_value = gt_prob
            
            print(f"  Ground truth probability: {gt_prob:.4f}")
            if use_log_prob:
                print(f"  Ground truth (log-space): {gt_value:.4f}")
            
            # GNN prediction - we need to process features
            # This is complex, so we'll do a simpler test
            print(f"\n  ⚠ Full feature test requires running benchmark")
            print(f"    Run: python benchmark_unified.py")
            print(f"    Expected MAE: <0.05 (good), 0.05-0.10 (acceptable), >0.10 (poor)")
    
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

# ===== SUMMARY =====
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

print("\n✓ If all checks passed:")
print("  1. Model architecture matches benchmark (25 features)")
print("  2. Normalization stats are present and valid")
print("  3. Config settings match training")
print("  → Ready to run benchmark!")

print("\n⚠ If you see warnings:")
print("  - Feature mismatch → Model will crash or give garbage")
print("  - Missing normalization → Predictions will be wrong")
print("  - Config mismatch → May give unexpected results")

print("\n📊 Expected Performance:")
print("  - Good GNN: MAE < 0.05, Time < 50ms")
print("  - Acceptable: MAE 0.05-0.10, Time < 100ms")
print("  - Poor: MAE > 0.10 or Time > 500ms")

print("\n🚀 Next Steps:")
print("  1. Fix any issues identified above")
print("  2. Run: python benchmark_unified.py")
print("  3. Check first few networks for MAE and timing")
print("  4. If MAE is high (>0.1), investigate:")
print("     - Is normalization being applied?")
print("     - Are features extracted correctly?")
print("     - Was model trained properly?")

print("\n" + "="*80)