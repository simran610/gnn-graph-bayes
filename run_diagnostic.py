"""
Run diagnostic to find EXACTLY why graphs fail 
"""

import torch
from bnlearn_diagnostic import BNLearnDiagnostic
from graphsage_model import GraphSAGE
from BIF_data_debugging import BenchmarkDatasetProcessor
import glob
import os
from pathlib import Path

def run_full_diagnostic():
    print("="*80)
    print("RUNNING BNLEARN DIAGNOSTIC")
    print("="*80)
    
    # Load model
    model_path = "training_results/models/graphsage_root_probability_evidence_only_intermediate_logprob_fold_4.pt"
    
    model = GraphSAGE(
        in_channels=25,
        hidden_channels=128,
        out_channels=1,
        mode='root_probability',
        use_log_prob=True
    )
    
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model = model.to('cuda')
    model.eval()
    print(f"✓ Loaded model from {model_path}")
    
    # Load BNLearn graphs
    processor = BenchmarkDatasetProcessor("config.yaml", verbose=False)
    bif_files = glob.glob("dataset_bif_files/*.bif")
    
    graphs = []
    print(f"\n✓ Loading {len(bif_files)} BNLearn graphs...")
    for bif_path in bif_files:
        network_name = Path(bif_path).stem
        try:
            graph, meta = processor.process_bif_to_graph(bif_path, network_name)
            if graph is not None:
                graph.name = network_name  # Add name attribute
                graphs.append(graph)
        except Exception as e:
            print(f"  ⚠️  Failed to load {network_name}: {e}")
    
    print(f"✓ Loaded {len(graphs)} graphs successfully")
    
    # Run diagnostic
    diagnostic = BNLearnDiagnostic(
        model=model,
        bnlearn_graphs=graphs,
        training_stats_path='datasets/folds/fold_0_norm_stats.pt',  # Training stats for comparison
        device='cuda'
    )
    
    results = diagnostic.diagnose_all()
    
    # Save report
    diagnostic.save_report(results, 'bnlearn_diagnostic_report.json')
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE!")
    print("="*80)
    print("✓ Report saved to: bnlearn_diagnostic_report.json")
    print("\nReview the 'root_causes' section to see exactly what's wrong!")
    
    return results

if __name__ == "__main__":
    results = run_full_diagnostic()