"""
Simple Safe Test - Works with ANY version of benchmark_unified.py
==================================================================
Tests GNN on small networks only to verify it works.
"""

import os
import sys
from pathlib import Path
from pgmpy.readwrite import BIFReader

print("="*80)
print("SIMPLE SAFE TEST")
print("="*80)

# Check files exist
required = ["model_finetuned_bnlearn.pt", "bnlearn_norm_stats.pt", "config.yaml"]
missing = [f for f in required if not Path(f).exists()]
if missing:
    print(f"❌ Missing files: {missing}")
    sys.exit(1)

# Find small BIF files only
bif_files = list(Path("dataset_bif_files").glob("*.bif"))
small_networks = []

print("\nFinding small networks (<100 nodes)...")
for bif in bif_files:
    try:
        reader = BIFReader(str(bif))
        model = reader.get_model()
        num_nodes = model.number_of_nodes()
        
        if num_nodes < 100:
            small_networks.append((bif, num_nodes))
            print(f"  ✓ {bif.stem}: {num_nodes} nodes")
    except:
        continue

small_networks = sorted(small_networks, key=lambda x: x[1])[:5]  # Take 5 smallest
print(f"\nTesting on {len(small_networks)} small networks")

# Now test with your original benchmark script
print("\n" + "="*80)
print("RUNNING TESTS")
print("="*80)

# Import your working benchmark
sys.path.insert(0, '.')
from BIF_data_debugging import ModelBenchmark, BenchmarkDatasetProcessor

processor = BenchmarkDatasetProcessor("config.yaml", verbose=True, cache_dir="cached_graphs")
benchmark = ModelBenchmark("model_finetuned_bnlearn.pt", "config.yaml")

graphs = []
metadata_list = []

for bif_path, num_nodes in small_networks:
    print(f"\nProcessing {bif_path.stem} ({num_nodes} nodes)...")
    try:
        graph, meta = processor.process_bif_to_graph(str(bif_path), bif_path.stem, use_cache=True)
        if graph is not None:
            graphs.append(graph)
            metadata_list.append(meta)
            print(f"  ✓ Success")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

if len(graphs) == 0:
    print("\n❌ No graphs processed!")
    sys.exit(1)

print(f"\n✓ Processed {len(graphs)} networks")

# Run predictions
print("\nRunning GNN predictions...")
results = benchmark.evaluate_dataset(graphs, metadata_list)

# Show results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
metrics = results["aggregate_metrics"]
print(f"MAE:  {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²:   {metrics.get('r2_score', 0):.4f}")

if "accuracy_within_10pct" in metrics:
    print(f"\nAccuracy within 10%: {metrics['accuracy_within_10pct']:.1%}")

per_network = results["per_network_results"]
print(f"\nPer-Network Results:")
for r in per_network:
    print(f"  {r['network_name']:15s} MAE: {r['absolute_error']:.4f}")

print("\n" + "="*80)
if metrics['mae'] < 0.20:
    print("✓ TEST PASSED! GNN is working")
    print("  MAE < 0.20 is acceptable")
    print("\nNext: Run full benchmark_unified.py")
else:
    print("⚠ MAE is high (>0.20)")
    print("  GNN may need more training or debugging")

print("="*80)