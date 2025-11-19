"""
BNLearn Failure Diagnostic Tool
Comprehensive diagnostic suite to identify root causes of failures on BNLearn benchmark graphs.
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader
import json
import os

class BNLearnDiagnostic:
    """Comprehensive diagnostic for BNLearn benchmark failures"""
    
    def __init__(self, model, bnlearn_graphs, training_stats_path, device='cuda'):
        self.model = model
        self.bnlearn_graphs = bnlearn_graphs
        self.device = device
        
        # Load training normalization stats
        if os.path.exists(training_stats_path):
            self.training_stats = torch.load(training_stats_path, weights_only=False)
            print(f"✓ Loaded training stats from {training_stats_path}")
        else:
            print(f"⚠️  Training stats not found at {training_stats_path}")
            self.training_stats = None
    
    def diagnose_all(self):
        """Run complete diagnostic suite"""
        print("="*80)
        print("BNLEARN DIAGNOSTIC SUITE")
        print("="*80 + "\n")
        
        results = {}
        
        # 1. Feature distribution analysis
        print("[1/6] Analyzing feature distributions...")
        results['feature_analysis'] = self.analyze_feature_distributions()
        
        # 2. Network structure analysis
        print("\n[2/6] Analyzing network structures...")
        results['structure_analysis'] = self.analyze_structures()
        
        # 3. Prediction analysis
        print("\n[3/6] Analyzing model predictions...")
        results['prediction_analysis'] = self.analyze_predictions()
        
        # 4. Activation analysis
        print("\n[4/6] Analyzing internal activations...")
        results['activation_analysis'] = self.analyze_activations()
        
        # 5. Normalization mismatch
        print("\n[5/6] Checking normalization mismatches...")
        results['normalization_analysis'] = self.check_normalization_mismatch()
        
        # 6. Root cause identification
        print("\n[6/6] Identifying root causes...")
        results['root_causes'] = self.identify_root_causes(results)
        
        return results
    
    def analyze_feature_distributions(self):
        """Compare BNLearn vs Training feature distributions"""
        print("  → Computing feature statistics...")
        
        # Collect all features from BNLearn graphs (handle CUDA tensors)
        all_features = []
        for graph in self.bnlearn_graphs:
            all_features.append(graph.x.cpu().numpy())  # FIX: Add .cpu()
        all_features = np.vstack(all_features)
        
        bnlearn_stats = {
            'mean': np.mean(all_features, axis=0),
            'std': np.std(all_features, axis=0),
            'min': np.min(all_features, axis=0),
            'max': np.max(all_features, axis=0),
            'has_nan': np.any(np.isnan(all_features)),
            'has_inf': np.any(np.isinf(all_features))
        }
        
        # Compare with training stats if available
        if self.training_stats:
            # Handle both dict and tensor formats
            if isinstance(self.training_stats, dict):
                train_mean = self.training_stats.get('mean', np.zeros_like(bnlearn_stats['mean']))
                train_std = self.training_stats.get('std', np.ones_like(bnlearn_stats['std']))
            else:
                # If it's a tensor or has base/cpd structure
                try:
                    train_mean = self.training_stats['base']['mean']
                    train_std = self.training_stats['base']['std']
                except:
                    train_mean = np.zeros_like(bnlearn_stats['mean'])
                    train_std = np.ones_like(bnlearn_stats['std'])
            
            # Compute divergence metrics
            mean_diff = np.abs(bnlearn_stats['mean'] - train_mean)
            std_ratio = bnlearn_stats['std'] / (train_std + 1e-8)
            
            bnlearn_stats['mean_divergence'] = mean_diff
            bnlearn_stats['std_ratio'] = std_ratio
            bnlearn_stats['large_divergence_features'] = np.where(mean_diff > 2.0)[0].tolist()
            bnlearn_stats['extreme_std_ratio'] = np.where((std_ratio < 0.5) | (std_ratio > 2.0))[0].tolist()
        
        print(f"  ✓ Features with NaN: {bnlearn_stats['has_nan']}")
        print(f"  ✓ Features with Inf: {bnlearn_stats['has_inf']}")
        if self.training_stats:
            print(f"  ✓ Features with large divergence: {len(bnlearn_stats.get('large_divergence_features', []))}")
        
        return bnlearn_stats
    
    def analyze_predictions(self):
        """Analyze what the model is actually predicting"""
        print("  → Running inference on BNLearn graphs...")
        
        self.model.eval()
        prediction_info = []
        
        with torch.no_grad():
            for i, graph in enumerate(self.bnlearn_graphs):
                graph = graph.to(self.device)
                
                # FIX: Add batch tensor
                graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
                
                try:
                    raw_output = self.model(graph)
                    # If model returns (out, aux) take main output
                    if isinstance(raw_output, (list, tuple)):
                        raw_main = raw_output[0]
                    else:
                        raw_main = raw_output
    
                    # Convert raw_main to numpy
                    if isinstance(raw_main, torch.Tensor):
                        raw_np = raw_main.detach().cpu().numpy()
                    elif isinstance(raw_main, np.ndarray):
                        raw_np = raw_main
                    else:
                        raw_np = np.array(raw_main)
    
                    # ground truth to numpy
                    gt = graph.y
                    if isinstance(gt, torch.Tensor):
                        gt_np = gt.detach().cpu().numpy()
                    elif isinstance(gt, np.ndarray):
                        gt_np = gt
                    else:
                        gt_np = np.array(gt)
    
                    # Make serializable-friendly values (scalars when possible)
                    def _maybe_scalar(a):
                        a = np.asarray(a)
                        if a.size == 1:
                            return float(a.item())
                        return a.tolist()
    
                    prediction_data = {
                        'graph_idx': i,
                        'name': getattr(graph, 'name', f'graph_{i}'),
                        'raw_output': _maybe_scalar(raw_np),
                        'ground_truth': _maybe_scalar(gt_np),
                        'error': _maybe_scalar(np.abs(raw_np - gt_np))
                    }
    
                    # warning on extreme scalar outputs (use first element if array-like)
                    try:
                        scalar_val = float(np.asarray(raw_np).flatten()[0])
                        if scalar_val < -50:
                            prediction_data['warning'] = f"Extreme negative output: {scalar_val:.2f}"
                        elif scalar_val > 10:
                            prediction_data['warning'] = f"Extreme positive output: {scalar_val:.2f}"
                    except Exception:
                        pass
    
                    prediction_info.append(prediction_data)
    
                except Exception as e:
                    prediction_info.append({
                        'graph_idx': i,
                        'name': getattr(graph, 'name', f'graph_{i}'),
                        'error_msg': str(e)
                    })
        
        # Identify catastrophic failures (prediction ≈ 0 when truth > 0.5)
        catastrophic = []
        for pred in prediction_info:
            if 'error_msg' in pred:
                continue
    
            raw_out = pred['raw_output']
            truth = pred['ground_truth']
    
            # Normalize to scalar floats for probability conversion
            def _to_scalar(v):
                if isinstance(v, list):
                    return float(v[0])
                try:
                    return float(v)
                except Exception:
                    # fallback if v is nested structure
                    arr = np.asarray(v)
                    return float(arr.flatten()[0]) if arr.size > 0 else None
    
            raw_val = _to_scalar(raw_out)
            truth_val = _to_scalar(truth)
            if raw_val is None or truth_val is None:
                continue
    
            # Convert from log-prob if negative (and fallback to model attribute if present)
            use_log = getattr(self.model, "use_log_prob", None)
            if use_log is True:
                pred_prob = float(np.exp(np.clip(raw_val, -50, 0)))
                truth_prob = float(np.exp(np.clip(truth_val, -50, 0)))
            else:
                pred_prob = float(np.exp(raw_val)) if raw_val < 0 else float(raw_val)
                truth_prob = float(np.exp(truth_val)) if truth_val < 0 else float(truth_val)
    
            if truth_prob > 0.5 and pred_prob < 0.001:
                catastrophic.append({
                    'graph_idx': pred['graph_idx'],
                    'name': pred['name'],
                    'raw_output': pred['raw_output'],
                    'pred_prob': pred_prob,
                    'ground_truth': pred['ground_truth'],
                    'truth_prob': truth_prob
                })
    
        print(f"  ✓ {len(prediction_info)} predictions made")
        print(f"  ✓ {len(catastrophic)} catastrophic failures detected")
        
        return {
            'all_predictions': prediction_info,
            'catastrophic_failures': catastrophic
        }
  
    def analyze_structures(self):
        """Analyze graph structural properties"""
        print("  → Analyzing graph structures...")
        
        structure_info = []
        
        for i, graph in enumerate(self.bnlearn_graphs):
            num_nodes = graph.x.shape[0]
            num_edges = graph.edge_index.shape[1]
            
            # Compute degree statistics
            edge_index = graph.edge_index.cpu()  # FIX: Move to CPU
            in_degrees = torch.zeros(num_nodes)
            out_degrees = torch.zeros(num_nodes)
            
            for edge in edge_index.t():
                source, target = edge[0].item(), edge[1].item()
                out_degrees[source] += 1
                in_degrees[target] += 1
            
            # Node type distribution
            node_types = graph.x[:, 0].cpu().numpy()  # FIX: Add .cpu()
            unique_types, type_counts = np.unique(node_types, return_counts=True)
            
            info = {
                'graph_idx': i,
                'name': getattr(graph, 'name', f'graph_{i}'),
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'avg_degree': (in_degrees.mean().item() + out_degrees.mean().item()) / 2,
                'max_in_degree': in_degrees.max().item(),
                'max_out_degree': out_degrees.max().item(),
                'node_type_distribution': dict(zip(unique_types.tolist(), type_counts.tolist())),
                'has_isolated_nodes': (in_degrees + out_degrees == 0).any().item(),
                'edge_density': num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
            }
            
            structure_info.append(info)
        
        print(f"  ✓ Analyzed {len(structure_info)} graphs")
        
        # Find structural outliers
        avg_degrees = [g['avg_degree'] for g in structure_info]
        densities = [g['edge_density'] for g in structure_info]
        
        print(f"  ✓ Avg degree range: [{min(avg_degrees):.2f}, {max(avg_degrees):.2f}]")
        print(f"  ✓ Density range: [{min(densities):.4f}, {max(densities):.4f}]")
        
        return structure_info
    
    def analyze_activations(self):
        """Analyze internal activations to find dead neurons"""
        print("  → Analyzing internal activations...")
        
        self.model.eval()
        activation_stats = {
            'layer1': {'dead_neurons': 0, 'low_activation_neurons': 0},
            'layer2': {'dead_neurons': 0, 'low_activation_neurons': 0},
            'layer3': {'dead_neurons': 0, 'low_activation_neurons': 0},
            'layer4': {'dead_neurons': 0, 'low_activation_neurons': 0}
        }
        
        # Hook to capture activations
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        hooks.append(self.model.sage1.register_forward_hook(get_activation('layer1')))
        hooks.append(self.model.sage2.register_forward_hook(get_activation('layer2')))
        hooks.append(self.model.sage3.register_forward_hook(get_activation('layer3')))
        hooks.append(self.model.sage4.register_forward_hook(get_activation('layer4')))
        
        with torch.no_grad():
            # Run on a few BNLearn graphs
            for graph in self.bnlearn_graphs[:5]:
                graph = graph.to(self.device)
                
                # FIX: Add batch tensor for model to work
                graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
                
                _ = self.model(graph)
                
                # Analyze activations
                for layer_name, act in activations.items():
                    act_np = act.cpu().numpy()  # FIX: Add .cpu()
                    
                    # Check for dead neurons (always zero)
                    dead = np.all(act_np == 0, axis=0).sum()
                    # Check for low activation (< 0.01 mean)
                    low_act = (np.abs(act_np).mean(axis=0) < 0.01).sum()
                    
                    activation_stats[layer_name]['dead_neurons'] = max(
                        activation_stats[layer_name]['dead_neurons'], dead
                    )
                    activation_stats[layer_name]['low_activation_neurons'] = max(
                        activation_stats[layer_name]['low_activation_neurons'], low_act
                    )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print(f"  ✓ Activation analysis complete")
        for layer, stats in activation_stats.items():
            if stats['dead_neurons'] > 0:
                print(f"  ⚠️  {layer}: {stats['dead_neurons']} dead neurons")
        
        return activation_stats
    
    def check_normalization_mismatch(self):
        """Check if normalization stats from training match BNLearn"""
        print("  → Checking normalization mismatch...")
        
        if not self.training_stats:
            return {'error': 'No training stats available'}
        
        # Sample features from BNLearn (FIX: move to CPU first)
        sample_features = []
        for graph in self.bnlearn_graphs[:10]:
            sample_features.append(graph.x.cpu().numpy())
        sample_features = np.vstack(sample_features)
        
        # Handle different training stats formats
        try:
            if isinstance(self.training_stats, dict) and 'base' in self.training_stats:
                # New format with base/cpd/distance
                train_mean = np.zeros(sample_features.shape[1])
                train_std = np.ones(sample_features.shape[1])
                
                # Fill in base features
                base_indices = self.training_stats['base']['indices']
                train_mean[base_indices] = self.training_stats['base']['mean']
                train_std[base_indices] = self.training_stats['base']['std']
            else:
                # Old format or simple dict
                train_mean = self.training_stats.get('mean', np.zeros(sample_features.shape[1]))
                train_std = self.training_stats.get('std', np.ones(sample_features.shape[1]))
        except Exception as e:
            print(f"  ⚠️  Error parsing training stats: {e}")
            train_mean = np.zeros(sample_features.shape[1])
            train_std = np.ones(sample_features.shape[1])
        
        bnlearn_mean = np.mean(sample_features, axis=0)
        bnlearn_std = np.std(sample_features, axis=0)
        
        # Compute normalized versions
        normalized_with_train = (sample_features - train_mean) / (train_std + 1e-8)
        normalized_with_bnlearn = (sample_features - bnlearn_mean) / (bnlearn_std + 1e-8)
        
        mismatch_score = np.mean(np.abs(normalized_with_train - normalized_with_bnlearn))
        
        result = {
            'mismatch_score': float(mismatch_score),
            'features_with_large_mismatch': np.where(
                np.abs(train_mean - bnlearn_mean) > 2.0 * train_std
            )[0].tolist(),
            'severe_mismatch': mismatch_score > 1.0
        }
        
        if result['severe_mismatch']:
            print(f"  ⚠️  SEVERE NORMALIZATION MISMATCH (score: {mismatch_score:.2f})")
        else:
            print(f"  ✓ Normalization mismatch score: {mismatch_score:.2f}")
        
        return result
    
    def identify_root_causes(self, analysis_results):
        """Synthesize all analyses to identify root causes"""
        print("  → Identifying root causes...")
        
        root_causes = []
        
        # Check 1: Normalization mismatch
        norm_analysis = analysis_results.get('normalization_analysis', {})
        if norm_analysis.get('severe_mismatch', False):
            root_causes.append({
                'cause': 'SEVERE_NORMALIZATION_MISMATCH',
                'severity': 'CRITICAL',
                'description': 'BNLearn features are normalized with training stats that dont match',
                'fix': 'Recompute normalization stats on BNLearn graphs or train with BNLearn data'
            })
        
        # Check 2: Extreme predictions
        pred_analysis = analysis_results.get('prediction_analysis', {})
        catastrophic = pred_analysis.get('catastrophic_failures', [])
        if len(catastrophic) > 5:
            root_causes.append({
                'cause': 'EXTREME_NEGATIVE_LOGPROB_OUTPUTS',
                'severity': 'CRITICAL',
                'description': f'{len(catastrophic)} graphs have log-prob < -20 causing underflow',
                'fix': 'Add output clamping: torch.clamp(output, min=-10, max=0)'
            })
        
        # Check 3: Dead neurons
        act_analysis = analysis_results.get('activation_analysis', {})
        total_dead = sum(layer['dead_neurons'] for layer in act_analysis.values())
        if total_dead > 10:
            root_causes.append({
                'cause': 'DEAD_NEURONS',
                'severity': 'HIGH',
                'description': f'{total_dead} dead neurons detected across layers',
                'fix': 'Lower learning rate, use leaky ReLU, or retrain model'
            })
        
        # Check 4: Feature distribution mismatch
        feat_analysis = analysis_results.get('feature_analysis', {})
        if len(feat_analysis.get('large_divergence_features', [])) > 5:
            root_causes.append({
                'cause': 'FEATURE_DISTRIBUTION_SHIFT',
                'severity': 'HIGH',
                'description': 'BNLearn graphs have very different feature distributions than training',
                'fix': 'Include BNLearn graphs in training or use domain adaptation'
            })
        
        # Check 5: Structural mismatch
        struct_analysis = analysis_results.get('structure_analysis', [])
        avg_bnlearn_degree = np.mean([g['avg_degree'] for g in struct_analysis])
        if avg_bnlearn_degree < 2.0 or avg_bnlearn_degree > 10.0:
            root_causes.append({
                'cause': 'STRUCTURAL_MISMATCH',
                'severity': 'MEDIUM',
                'description': f'BNLearn graphs have unusual structure (avg degree: {avg_bnlearn_degree:.2f})',
                'fix': 'Generate training graphs with similar structure to BNLearn'
            })
        
        print(f"\n  {'='*76}")
        print(f"  IDENTIFIED {len(root_causes)} ROOT CAUSES")
        print(f"  {'='*76}")
        for i, cause in enumerate(root_causes, 1):
            print(f"\n  [{i}] {cause['cause']} (Severity: {cause['severity']})")
            print(f"      Description: {cause['description']}")
            print(f"      Fix: {cause['fix']}")
        
        return root_causes
    
    def save_report(self, results, output_path='bnlearn_diagnostic_report.json'):
        """Save diagnostic report to file"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):  
                return bool(obj)  
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✓ Diagnostic report saved to {output_path}")


if __name__ == "__main__":
    print("BNLearn Diagnostic Tool - Import and use in run_diagnostic.py")