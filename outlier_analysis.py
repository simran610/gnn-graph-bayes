"""
Outlier Analysis Module

Detects and analyzes anomalous predictions from GNN models. Identifies problematic
predictions, graphs, and features. Provides statistical insights into outlier
characteristics and visualization tools.

Main Functions:
    - analyze_outliers: Comprehensive outlier detection and statistical analysis
    - run_outlier_analysis_for_gnn: Full analysis pipeline for trained models
    - analyze_graph_structure_outliers: Detect outliers based on graph properties
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances
import torch
import torch.nn.functional as F
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


def analyze_outliers(y_true, y_pred, X_features, graph_data=None, feature_names=None, outlier_percentile=95):
    """
    Comprehensive outlier analysis for GNN predictions
    """
    # 1. EXTRACT OUTLIERS
    residuals = np.abs(y_true - y_pred)
    outlier_threshold = np.percentile(residuals, outlier_percentile)
    outlier_mask = residuals > outlier_threshold
    outlier_indices = np.where(outlier_mask)[0]
    
    print(f"Found {len(outlier_indices)} outliers ({len(outlier_indices)/len(y_true)*100:.1f}%)")
    print(f"Outlier threshold: {outlier_threshold:.4f}")
    
    # 2. BASIC STATISTICS
    print("\n=== OUTLIER STATISTICS ===")
    print(f"Mean residual (normal): {residuals[~outlier_mask].mean():.4f}")
    print(f"Mean residual (outliers): {residuals[outlier_mask].mean():.4f}")
    print(f"Max residual: {residuals.max():.4f}")
    
    # 3. PROBABILITY RANGE ANALYSIS
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # True vs Predicted with outliers highlighted
    axes[0,0].scatter(y_true[~outlier_mask], y_pred[~outlier_mask], alpha=0.6, label='Normal', s=20)
    axes[0,0].scatter(y_true[outlier_mask], y_pred[outlier_mask], color='red', label='Outliers', s=30)
    axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0,0].set_xlabel('True Probability')
    axes[0,0].set_ylabel('Predicted Probability')
    axes[0,0].legend()
    axes[0,0].set_title('True vs Predicted (Outliers Highlighted)')
    
    # Residuals vs True Values
    axes[0,1].scatter(y_true, residuals, alpha=0.6)
    axes[0,1].axhline(outlier_threshold, color='red', linestyle='--', label=f'{outlier_percentile}th percentile')
    axes[0,1].set_xlabel('True Probability')
    axes[0,1].set_ylabel('Absolute Residual')
    axes[0,1].legend()
    axes[0,1].set_title('Residuals vs True Values')
    
    # Probability distribution comparison
    axes[1,0].hist(y_true[~outlier_mask], bins=20, alpha=0.7, label='Normal', density=True)
    axes[1,0].hist(y_true[outlier_mask], bins=20, alpha=0.7, label='Outliers', density=True)
    axes[1,0].set_xlabel('True Probability')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    axes[1,0].set_title('Probability Distribution Comparison')
    
    # Residual distribution
    axes[1,1].hist(residuals[~outlier_mask], bins=30, alpha=0.7, label='Normal')
    axes[1,1].hist(residuals[outlier_mask], bins=15, alpha=0.7, label='Outliers')
    axes[1,1].set_xlabel('Absolute Residual')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    axes[1,1].set_title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('outlier_analysis_plots.png')
    
    # 4. FEATURE ANALYSIS
    if X_features is not None:
        n_features = X_features.shape[1]
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
            
        print("\n=== FEATURE ANALYSIS ===")
        
        # Statistical tests for each feature
        outlier_features = []
        for i in range(n_features):
            normal_vals = X_features[~outlier_mask, i]
            outlier_vals = X_features[outlier_mask, i]
            
            # Mann-Whitney U test
            try:
                statistic, p_value = stats.mannwhitneyu(normal_vals, outlier_vals)
                if p_value < 0.05:
                    outlier_features.append((i, feature_names[i], p_value))
            except:
                pass
        
        print(f"Features significantly different in outliers (p < 0.05):")
        for feat_idx, feat_name, p_val in sorted(outlier_features, key=lambda x: x[2]):
            print(f"  {feat_name}: p = {p_val:.4f}")
    
    # 5. PROBABILITY RANGE OUTLIERS
    print("\n=== PROBABILITY RANGE ANALYSIS ===")
    extreme_low = (y_true < 0.1) & outlier_mask
    extreme_high = (y_true > 0.9) & outlier_mask  
    mid_range = ((y_true >= 0.1) & (y_true <= 0.9)) & outlier_mask
    
    print(f"Outliers in extreme low range (0-0.1): {extreme_low.sum()}")
    print(f"Outliers in extreme high range (0.9-1.0): {extreme_high.sum()}")
    print(f"Outliers in mid range (0.1-0.9): {mid_range.sum()}")
    
    # 6. SPECIFIC OUTLIER ANALYSIS
    pick_specific_prediction_outliers(y_true, y_pred, X_features, feature_names, n_each=2)
    
    return {
        'indices': outlier_indices,
        'residuals': residuals[outlier_mask],
        'true_values': y_true[outlier_mask],
        'pred_values': y_pred[outlier_mask],
        'threshold': outlier_threshold
    }


def pick_specific_prediction_outliers(y_true, y_pred, X_features, feature_names, n_each=2):
    """
    Pick specific over-predictions and under-predictions far from diagonal
    """
    residuals = y_pred - y_true  # Signed residuals
    abs_residuals = np.abs(residuals)
    
    # Over-predictions (predicted > true, far from diagonal)
    over_pred_mask = residuals > 0
    over_pred_distances = abs_residuals[over_pred_mask]
    over_pred_indices = np.where(over_pred_mask)[0]
    
    # Under-predictions (predicted < true, far from diagonal) 
    under_pred_mask = residuals < 0
    under_pred_distances = abs_residuals[under_pred_mask]
    under_pred_indices = np.where(under_pred_mask)[0]
    
    # Pick worst cases
    worst_over = over_pred_indices[np.argsort(over_pred_distances)[-n_each:]]
    worst_under = under_pred_indices[np.argsort(under_pred_distances)[-n_each:]]
    
    print("\n=== SPECIFIC OUTLIER ANALYSIS ===")
    
    # Analyze over-predictions
    print(f"\n{n_each} Worst OVER-predictions:")
    for i, idx in enumerate(worst_over):
        print(f"Sample {idx}: True={y_true[idx]:.3f}, Pred={y_pred[idx]:.3f}, Error={residuals[idx]:.3f}")
        
        if X_features is not None:
            features = X_features[idx]
            print(f"  Evidence nodes: {int(features[9])}, Num parents: {int(features[8])}")
            print(f"  In/Out degree: {int(features[1])}/{int(features[2])}")
            print(f"  Betweenness: {features[3]:.3f}, PageRank: {features[5]:.3f}")
    
    # Analyze under-predictions  
    print(f"\n{n_each} Worst UNDER-predictions:")
    for i, idx in enumerate(worst_under):
        print(f"Sample {idx}: True={y_true[idx]:.3f}, Pred={y_pred[idx]:.3f}, Error={residuals[idx]:.3f}")
        
        if X_features is not None:
            features = X_features[idx]
            print(f"  Evidence nodes: {int(features[9])}, Num parents: {int(features[8])}")
            print(f"  In/Out degree: {int(features[1])}/{int(features[2])}")
            print(f"  Betweenness: {features[3]:.3f}, PageRank: {features[5]:.3f}")
    
    return {
        'over_predictions': worst_over,
        'under_predictions': worst_under,
        'over_errors': residuals[worst_over],
        'under_errors': residuals[worst_under]
    }


def analyze_graph_structure_outliers(model, test_loader, device, mode="root_probability"):
    """
    Analyze outliers based on graph structure rather than just node features
    """
    model.eval()
    outlier_data = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            out = model(data)
            
            # Get predictions and targets
            if mode == "root_probability":
                targets = data.y.squeeze()
                if isinstance(out, tuple):
                    out = out[0]
                predictions = out.squeeze()
            
            # Convert to numpy
            if predictions.dim() == 0:
                preds = predictions.cpu().numpy().reshape(1)
                trues = targets.cpu().numpy().reshape(1)
            else:
                preds = predictions.cpu().numpy()
                trues = targets.cpu().numpy()
            
            # Analyze each graph in the batch
            batch_size = data.batch.max().item() + 1
            for i in range(batch_size):
                mask = data.batch == i
                graph_nodes = torch.where(mask)[0]
                
                # Extract subgraph
                node_features = data.x[mask]
                
                # Find edges for this subgraph
                edge_mask = (data.edge_index[0].unsqueeze(1) == graph_nodes).any(1) & \
                           (data.edge_index[1].unsqueeze(1) == graph_nodes).any(1)
                subgraph_edges = data.edge_index[:, edge_mask]
                
                # Remap edge indices to local numbering
                node_mapping = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(graph_nodes)}
                local_edges = torch.tensor([[node_mapping[subgraph_edges[0][j].item()], 
                                           node_mapping[subgraph_edges[1][j].item()]] 
                                          for j in range(subgraph_edges.shape[1])]).T
                
                # Calculate graph-level features
                graph_features = extract_graph_level_features(node_features, local_edges, i)
                
                # Store data
                pred_val = preds[i] if preds.ndim > 0 else preds
                true_val = trues[i] if trues.ndim > 0 else trues
                
                outlier_data.append({
                    'graph_id': f"batch_{batch_idx}_graph_{i}",
                    'true_prob': true_val,
                    'pred_prob': pred_val,
                    'error': abs(pred_val - true_val),
                    'signed_error': pred_val - true_val,
                    **graph_features
                })
    
    return analyze_structural_patterns(outlier_data)


def extract_graph_level_features(node_features, edge_index, graph_id):
    """
    Extract meaningful graph-level structural features
    """
    num_nodes = node_features.shape[0]
    
    # Convert to NetworkX for analysis
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    if edge_index.shape[1] > 0:
        edges = edge_index.T.numpy()
        G.add_edges_from(edges)
    
    # Root node identification (assuming node 0 is root)
    root_idx = 0
    
    # Graph topology features
    features = {
        # Basic structure
        'num_nodes': num_nodes,
        'num_edges': edge_index.shape[1],
        'graph_density': edge_index.shape[1] / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
        
        # Evidence distribution
        'num_evidence_nodes': (node_features[:, 9] == 1).sum().item(),
        'evidence_ratio': (node_features[:, 9] == 1).sum().item() / num_nodes,
        
        # Distance from root to evidence
        'min_root_to_evidence_dist': get_min_distance_to_evidence(G, root_idx, node_features),
        'max_root_to_evidence_dist': get_max_distance_to_evidence(G, root_idx, node_features),
        'avg_root_to_evidence_dist': get_avg_distance_to_evidence(G, root_idx, node_features),
        
        # Graph shape characteristics
        'max_depth': get_graph_depth(G, root_idx),
        'branching_factor': get_avg_branching_factor(G),
        'leaf_nodes_count': sum(1 for n in G.nodes() if G.out_degree(n) == 0),
        
        # Evidence clustering
        'evidence_clustering': get_evidence_clustering(G, node_features),
        
        # Path characteristics
        'longest_path_length': get_longest_path_from_root(G, root_idx),
        'num_root_children': G.out_degree(root_idx),
        
        # CPD characteristics in evidence nodes
        'evidence_cpd_entropy': get_evidence_cpd_entropy(node_features),
        'evidence_cpd_max': get_evidence_cpd_stats(node_features, 'max'),
        'evidence_cpd_min': get_evidence_cpd_stats(node_features, 'min'),
    }
    
    return features


def get_min_distance_to_evidence(G, root_idx, node_features):
    """Get minimum distance from root to any evidence node"""
    evidence_nodes = [i for i in range(node_features.shape[0]) if node_features[i, 9] == 1]
    if not evidence_nodes:
        return float('inf')
    
    distances = []
    for ev_node in evidence_nodes:
        try:
            dist = nx.shortest_path_length(G, root_idx, ev_node)
            distances.append(dist)
        except nx.NetworkXNoPath:
            distances.append(float('inf'))
    
    return min(distances) if distances else float('inf')


def get_max_distance_to_evidence(G, root_idx, node_features):
    """Get maximum distance from root to any evidence node"""
    evidence_nodes = [i for i in range(node_features.shape[0]) if node_features[i, 9] == 1]
    if not evidence_nodes:
        return 0
    
    distances = []
    for ev_node in evidence_nodes:
        try:
            dist = nx.shortest_path_length(G, root_idx, ev_node)
            distances.append(dist)
        except nx.NetworkXNoPath:
            distances.append(0)
    
    return max(distances) if distances else 0


def get_avg_distance_to_evidence(G, root_idx, node_features):
    """Get average distance from root to evidence nodes"""
    evidence_nodes = [i for i in range(node_features.shape[0]) if node_features[i, 9] == 1]
    if not evidence_nodes:
        return float('inf')
    
    distances = []
    for ev_node in evidence_nodes:
        try:
            dist = nx.shortest_path_length(G, root_idx, ev_node)
            distances.append(dist)
        except nx.NetworkXNoPath:
            pass
    
    return sum(distances) / len(distances) if distances else float('inf')


def get_graph_depth(G, root_idx):
    """Get maximum depth of the graph from root"""
    try:
        lengths = nx.single_source_shortest_path_length(G, root_idx)
        return max(lengths.values()) if lengths else 0
    except:
        return 0


def get_avg_branching_factor(G):
    """Get average branching factor"""
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    return sum(out_degrees) / len(out_degrees) if out_degrees else 0


def get_evidence_clustering(G, node_features):
    """Measure how clustered evidence nodes are"""
    evidence_nodes = [i for i in range(node_features.shape[0]) if node_features[i, 9] == 1]
    if len(evidence_nodes) < 2:
        return 0
    
    # Count edges between evidence nodes
    evidence_edges = 0
    for i in evidence_nodes:
        for j in evidence_nodes:
            if i != j and G.has_edge(i, j):
                evidence_edges += 1
    
    # Normalize by possible edges
    max_possible = len(evidence_nodes) * (len(evidence_nodes) - 1)
    return evidence_edges / max_possible if max_possible > 0 else 0


def get_longest_path_from_root(G, root_idx):
    """Get length of longest path from root"""
    try:
        if G.is_directed() and nx.is_directed_acyclic_graph(G):
            leaf_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
            if leaf_nodes:
                paths = []
                for leaf in leaf_nodes:
                    try:
                        path_len = nx.shortest_path_length(G, root_idx, leaf)
                        paths.append(path_len)
                    except nx.NetworkXNoPath:
                        pass
                return max(paths) if paths else 0
        return get_graph_depth(G, root_idx)
    except:
        return get_graph_depth(G, root_idx)


def get_evidence_cpd_entropy(node_features):
    """Calculate entropy of CPD values in evidence nodes"""
    evidence_mask = node_features[:, 9] == 1
    if evidence_mask.sum() == 0:
        return 0
    
    evidence_cpds = node_features[evidence_mask, 10:18]  # CPD features
    entropies = []
    
    for cpd_row in evidence_cpds:
        # Normalize to probabilities
        cpd_probs = torch.softmax(cpd_row, dim=0)
        # Calculate entropy
        entropy = -torch.sum(cpd_probs * torch.log(cpd_probs + 1e-8))
        entropies.append(entropy.item())
    
    return np.mean(entropies) if entropies else 0


def get_evidence_cpd_stats(node_features, stat_type):
    """Get statistics of CPD values in evidence nodes"""
    evidence_mask = node_features[:, 9] == 1
    if evidence_mask.sum() == 0:
        return 0
    
    evidence_cpds = node_features[evidence_mask, 10:18]
    
    if stat_type == 'max':
        return evidence_cpds.max().item()
    elif stat_type == 'min':
        return evidence_cpds.min().item()
    else:
        return evidence_cpds.mean().item()


def analyze_structural_patterns(outlier_data):
    """
    Analyze patterns in graph structure that correlate with prediction errors
    """
    # Convert to DataFrame for analysis
    import pandas as pd
    df = pd.DataFrame(outlier_data)
    
    # Define outliers as top 5% of errors
    error_threshold = df['error'].quantile(0.95)
    df['is_outlier'] = df['error'] > error_threshold
    
    # Define severe underpredictions
    df['severe_underpredict'] = (df['signed_error'] < -0.2) & (df['true_prob'] > 0.7)
    
    print("=== GRAPH STRUCTURE OUTLIER ANALYSIS ===")
    print(f"Total graphs analyzed: {len(df)}")
    print(f"Outliers (top 5% errors): {df['is_outlier'].sum()}")
    print(f"Severe underpredictions: {df['severe_underpredict'].sum()}")
    
    # Analyze structural differences
    structural_features = [col for col in df.columns if col not in 
                          ['graph_id', 'true_prob', 'pred_prob', 'error', 'signed_error', 'is_outlier', 'severe_underpredict']]
    
    print("\n=== STRUCTURAL PATTERNS IN OUTLIERS ===")
    for feature in structural_features:
        outlier_vals = df[df['is_outlier']][feature]
        normal_vals = df[~df['is_outlier']][feature]
        
        if len(outlier_vals) > 0 and len(normal_vals) > 0:
            # Statistical test
            from scipy.stats import mannwhitneyu
            try:
                stat, p_val = mannwhitneyu(outlier_vals, normal_vals)
                if p_val < 0.05:
                    print(f"{feature}:")
                    print(f"  Outliers: mean={outlier_vals.mean():.3f}, std={outlier_vals.std():.3f}")
                    print(f"  Normal: mean={normal_vals.mean():.3f}, std={normal_vals.std():.3f}")
                    print(f"  p-value: {p_val:.4f}")
            except:
                pass
    
    print("\n=== SEVERE UNDERPREDICTION PATTERNS ===")
    if df['severe_underpredict'].sum() > 0:
        severe_cases = df[df['severe_underpredict']]
        print("Common characteristics in severe underpredictions:")
        for feature in structural_features:
            vals = severe_cases[feature]
            if len(vals) > 0:
                print(f"  {feature}: mean={vals.mean():.3f}, std={vals.std():.3f}")
    
    # Visualization
    plot_structural_analysis(df, structural_features)
    
    return df


def plot_structural_analysis(df, structural_features):
    """Create plots showing structural patterns"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot key structural features
    key_features = ['min_root_to_evidence_dist', 'evidence_ratio', 'max_depth', 
                   'num_evidence_nodes', 'branching_factor', 'evidence_cpd_entropy']
    
    for i, feature in enumerate(key_features[:6]):
        if feature in df.columns:
            axes[i].scatter(df[feature], df['error'], alpha=0.6, s=20)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Prediction Error')
            axes[i].set_title(f'Error vs {feature}')
            
            # Highlight severe underpredictions
            severe = df[df['severe_underpredict']]
            if len(severe) > 0:
                axes[i].scatter(severe[feature], severe['error'], 
                               color='red', s=30, alpha=0.8, label='Severe Underpredict')
                axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('structural_outlier_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Additional detailed plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # True vs Predicted with structural coloring
    if 'min_root_to_evidence_dist' in df.columns:
        scatter = axes[0,0].scatter(df['true_prob'], df['pred_prob'], 
                                   c=df['min_root_to_evidence_dist'], 
                                   cmap='viridis', alpha=0.6, s=20)
        axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0,0].set_xlabel('True Probability')
        axes[0,0].set_ylabel('Predicted Probability')
        axes[0,0].set_title('Predictions colored by Root-to-Evidence Distance')
        plt.colorbar(scatter, ax=axes[0,0])
    
    # Error vs Evidence Ratio
    if 'evidence_ratio' in df.columns:
        axes[0,1].scatter(df['evidence_ratio'], df['signed_error'], alpha=0.6, s=20)
        axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0,1].set_xlabel('Evidence Ratio')
        axes[0,1].set_ylabel('Signed Error (Pred - True)')
        axes[0,1].set_title('Prediction Bias vs Evidence Ratio')
    
    # Depth vs Error
    if 'max_depth' in df.columns:
        axes[1,0].scatter(df['max_depth'], df['error'], alpha=0.6, s=20)
        axes[1,0].set_xlabel('Graph Depth')
        axes[1,0].set_ylabel('Absolute Error')
        axes[1,0].set_title('Error vs Graph Depth')
    
    # Evidence nodes vs Error  
    if 'num_evidence_nodes' in df.columns:
        axes[1,1].scatter(df['num_evidence_nodes'], df['signed_error'], alpha=0.6, s=20)
        axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1,1].set_xlabel('Number of Evidence Nodes')
        axes[1,1].set_ylabel('Signed Error')
        axes[1,1].set_title('Prediction Bias vs Evidence Count')
    
    plt.tight_layout()
    plt.savefig('structural_patterns_detailed.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_outlier_analysis_for_gnn(model, test_loader, device, mode="root_probability"):
    """
    Wrapper function to run outlier analysis on GNN test results
    """
    model.eval()
    all_preds = []
    all_trues = []
    all_features = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            
            # Extract predictions and targets based on mode
            if mode == "root_probability":
                targets = data.y.squeeze()
                if isinstance(out, tuple):
                    out = out[0]
                predictions = out.squeeze()
            elif mode == "distribution":
                batch_size = data.batch.max().item() + 1
                targets = data.y.view(batch_size, 2)
                predictions = F.softmax(out, dim=1)
            else:  # regression
                targets = data.y
                predictions = out
            
            # Collect data
            if predictions.dim() == 0:
                all_preds.append(predictions.cpu().unsqueeze(0))
            else:
                all_preds.append(predictions.cpu())
                
            if targets.dim() == 0:
                all_trues.append(targets.cpu().unsqueeze(0))
            else:
                all_trues.append(targets.cpu())
            
            # Extract ROOT NODE features for each graph
            batch_size = data.batch.max().item() + 1
            for i in range(batch_size):
                mask = data.batch == i
                graph_nodes = torch.where(mask)[0]
                
                # Assume root node is the first node in each graph
                root_idx = graph_nodes[0]
                root_features = data.x[root_idx]
                all_features.append(root_features.cpu())
    
    # Convert to numpy
    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_trues).numpy()
    features = torch.stack(all_features).numpy()
    
    # For root_probability mode, take probability of class 1
    if mode == "root_probability" and preds.ndim == 1:
        pass
    elif mode == "distribution":
        trues = trues[:, 1]
        preds = preds[:, 1]
    
    feature_names = [
        'node_type',         # 0
        'in_degree',         # 1
        'out_degree',        # 2
        'betweenness',       # 3
        'closeness',         # 4
        'pagerank',          # 5
        'degree_centrality', # 6
        'variable_card',     # 7
        'num_parents',       # 8
        'evidence_flag',     # 9
        'cpd_0',             # 10
        'cpd_1',             # 11
        'cpd_2',             # 12
        'cpd_3',             # 13
        'cpd_4',             # 14
        'cpd_5',             # 15
        'cpd_6',             # 16
        'cpd_7'              # 17
    ]
    
    print("=== RUNNING OUTLIER ANALYSIS ===")
    print(f"Sample root features shape: {features[0].shape}")
    print(f"First root node features: {features[0][:5]}")
    
    outlier_info = analyze_outliers(
        y_true=trues,
        y_pred=preds,
        X_features=features,
        feature_names=feature_names,
        outlier_percentile=95
    )
    
    return outlier_info


# Usage examples:
# 1. Basic outlier analysis (what you're currently doing):
# outlier_info = run_outlier_analysis_for_gnn(model, test_loader, device)

# 2. Advanced graph structure analysis (recommended):  
# structural_analysis = analyze_graph_structure_outliers(model, test_loader, device)