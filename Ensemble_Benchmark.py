"""
Ensemble Benchmark - Test multiple folds and average results
Works with both log-prob and normal-prob modes
"""

import torch
import numpy as np
import json
import os
from benchmark_truly_fixed import (
    ModelBenchmark, 
    BenchmarkDatasetProcessor,
    compute_rootprob_advanced_metrics
)
import glob
from pathlib import Path
import matplotlib.pyplot as plt


def run_ensemble_benchmark(config_path="config.yaml", 
                           fold_pattern="training_results/models/*fold_*.pt",
                           bif_directory="dataset_bif_files",
                           output_dir="benchmark_results/ensemble"):
    
    print("=" * 70)
    print("ENSEMBLE BENCHMARK (Multiple Folds)")
    print("=" * 70)
    
    # Find all fold models
    model_paths = sorted(glob.glob(fold_pattern))
    print(f"\nFound {len(model_paths)} fold models:")
    for path in model_paths:
        print(f"  - {Path(path).name}")
    
    if len(model_paths) == 0:
        print("❌ No models found!")
        print(f"   Pattern: {fold_pattern}")
        return
    
    # Process graphs once
    print("\n[1/3] Loading benchmark graphs...")
    processor = BenchmarkDatasetProcessor(config_path, verbose=False)
    bif_files = sorted(glob.glob(os.path.join(bif_directory, "*.bif")))
    
    graphs, metadata_list = [], []
    for bif_path in bif_files:
        network_name = Path(bif_path).stem
        try:
            graph, meta = processor.process_bif_to_graph(bif_path, network_name, use_cache=True)
            if graph is not None:  # Skip filtered networks
                graphs.append(graph)
                metadata_list.append(meta)
        except Exception as e:
            print(f"✗ {network_name}: {e}")
    
    print(f"✓ Loaded {len(graphs)} graphs")
    
    # Run each fold
    print("\n[2/3] Running predictions on all folds...")
    all_fold_predictions = []
    fold_names = []
    
    for fold_idx, model_path in enumerate(model_paths):
        fold_name = Path(model_path).stem
        fold_names.append(fold_name)
        print(f"\n  Fold {fold_idx + 1}/{len(model_paths)}: {fold_name}")
        
        try:
            benchmark = ModelBenchmark(model_path, config_path)
            fold_preds = []
            
            for graph in graphs:
                pred = benchmark.predict_single_graph(graph)
                fold_preds.append(pred)
            
            all_fold_predictions.append(fold_preds)
            
            # Convert to prob-space for display
            if processor.use_log_prob:
                display_preds = np.exp(np.clip(fold_preds, -10, 0))
            else:
                display_preds = fold_preds
            
            print(f"    ✓ Predictions range: [{np.min(display_preds):.4f}, {np.max(display_preds):.4f}]")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            continue
    
    if len(all_fold_predictions) == 0:
        print("❌ No successful predictions!")
        return
    
    # Ensemble predictions (average)
    print("\n[3/3] Computing ensemble results...")
    all_fold_predictions = np.array(all_fold_predictions)  # Shape: (n_folds, n_graphs)
    ensemble_preds = np.mean(all_fold_predictions, axis=0)
    ensemble_std = np.std(all_fold_predictions, axis=0)
    
    # Get ground truths
    ground_truths = np.array([g.y.item() for g in graphs])
    
    # Compute metrics
    use_log_prob = processor.use_log_prob
    metrics = compute_rootprob_advanced_metrics(ensemble_preds, ground_truths, use_log_prob)
    
    # Convert to prob-space for display
    if use_log_prob:
        preds_prob = np.exp(np.clip(ensemble_preds, -10, 0))
        trues_prob = np.exp(np.clip(ground_truths, -10, 0))
        stds_prob = ensemble_std  # std in log-space
    else:
        preds_prob = ensemble_preds
        trues_prob = ground_truths
        stds_prob = ensemble_std
    
    # Print results
    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)
    
    print(f"\nEnsemble Configuration:")
    print(f"  Number of folds: {len(all_fold_predictions)}")
    print(f"  Mode: {'log-probability' if use_log_prob else 'probability'}")
    print(f"  Networks tested: {len(graphs)}")
    
    print(f"\nCore Metrics (Probability Space):")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2_score']:.4f}")
    
    if 'mae_logspace' in metrics:
        print(f"\nLog-Space Metrics:")
        print(f"  MAE:  {metrics['mae_logspace']:.4f}")
        print(f"  RMSE: {metrics['rmse_logspace']:.4f}")
    
    print(f"\nTolerance Accuracy:")
    print(f"  Within  5%: {metrics.get('accuracy_within_5pct', 0):.2%}")
    print(f"  Within 10%: {metrics.get('accuracy_within_10pct', 0):.2%}")
    print(f"  Within 15%: {metrics.get('accuracy_within_15pct', 0):.2%}")
    
    print(f"\nEnsemble Uncertainty:")
    print(f"  Mean prediction std: {np.mean(stds_prob):.4f}")
    print(f"  Max prediction std:  {np.max(stds_prob):.4f}")
    print(f"  Min prediction std:  {np.min(stds_prob):.4f}")
    
    # Per-network results with uncertainty
    per_network_results = []
    for i, meta in enumerate(metadata_list):
        per_network_results.append({
            "network_name": meta["network_name"],
            "num_nodes": meta["num_nodes"],
            "ground_truth_prob": trues_prob[i],
            "ensemble_prediction_prob": preds_prob[i],
            "prediction_std": stds_prob[i],
            "absolute_error": abs(preds_prob[i] - trues_prob[i]),
            "fold_predictions": [float(all_fold_predictions[j, i]) for j in range(len(all_fold_predictions))]
        })
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "ensemble_metrics": metrics,
        "num_folds": len(all_fold_predictions),
        "use_log_prob": use_log_prob,
        "predictions_mean": ensemble_preds.tolist(),
        "predictions_std": ensemble_std.tolist(),
        "ground_truths": ground_truths.tolist(),
        "per_network_results": per_network_results,
        "fold_names": fold_names
    }
    
    with open(os.path.join(output_dir, "ensemble_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Visualizations
    print("\n[4/4] Creating visualizations...")
    
    # 1. Scatter plot with error bars
    plt.figure(figsize=(10, 10))
    plt.errorbar(trues_prob, preds_prob, yerr=stds_prob, fmt='o', alpha=0.6, 
                 elinewidth=1, capsize=3, markersize=8, label='Ensemble ± std')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True Probability', fontsize=12)
    plt.ylabel('Predicted Probability', fontsize=12)
    plt.title(f'Ensemble Predictions ({len(all_fold_predictions)} folds)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_scatter.png'), dpi=300)
    plt.close()
    
    # 2. Uncertainty vs Error
    errors = np.abs(preds_prob - trues_prob)
    plt.figure(figsize=(10, 6))
    plt.scatter(stds_prob, errors, alpha=0.6, s=100)
    plt.xlabel('Prediction Std Dev', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Ensemble Uncertainty vs Prediction Error', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_vs_error.png'), dpi=300)
    plt.close()
    
    # 3. Per-fold performance comparison
    fold_maes = []
    for fold_preds in all_fold_predictions:
        if use_log_prob:
            fold_preds_prob = np.exp(np.clip(fold_preds, -10, 0))
        else:
            fold_preds_prob = fold_preds
        fold_mae = np.mean(np.abs(fold_preds_prob - trues_prob))
        fold_maes.append(fold_mae)
    
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(fold_maes))
    plt.bar(x_pos, fold_maes, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axhline(y=metrics['mae'], color='red', linestyle='--', linewidth=2, 
                label=f'Ensemble MAE: {metrics["mae"]:.4f}')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Per-Fold Performance', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, [f'Fold {i+1}' for i in range(len(fold_maes))], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_fold_performance.png'), dpi=300)
    plt.close()
    
    print(f"\n✓ Saved results to {output_dir}/")
    print(f"  - ensemble_results.json")
    print(f"  - ensemble_scatter.png")
    print(f"  - uncertainty_vs_error.png")
    print(f"  - per_fold_performance.png")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ensemble benchmark on multiple folds")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--pattern", default="training_results/models/*fold_*.pt", 
                       help="Glob pattern for fold models")
    parser.add_argument("--bif-dir", default="dataset_bif_files", 
                       help="Directory with BIF files")
    parser.add_argument("--output", default="benchmark_results/ensemble", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    run_ensemble_benchmark(
        config_path=args.config,
        fold_pattern=args.pattern,
        bif_directory=args.bif_dir,
        output_dir=args.output
    )