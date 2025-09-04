import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

def analyze_outliers(y_true, y_pred, X_features, graph_data=None, feature_names=None, outlier_percentile=95):
    """
    Comprehensive outlier analysis for GNN predictions
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values  
        X_features: Node features (n_samples, n_features)
        graph_data: Optional graph properties dict
        feature_names: List of feature names
        outlier_percentile: Percentile threshold for outliers
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
    #plt.show()
    plt.savefig('outlier_analysis_plots.png')  # Save figure to file for ssh remote access
    
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
        
        # Feature boxplots for top different features
        if outlier_features:
            n_plots = min(6, len(outlier_features))
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            for i, (feat_idx, feat_name, _) in enumerate(outlier_features[:n_plots]):
                data_normal = X_features[~outlier_mask, feat_idx]
                data_outlier = X_features[outlier_mask, feat_idx]
                
                axes[i].boxplot([data_normal, data_outlier], labels=['Normal', 'Outliers'])
                axes[i].set_title(f'{feat_name}')
                axes[i].set_ylabel('Feature Value')
            
            # Hide empty subplots
            for i in range(n_plots, 6):
                axes[i].axis('off')
                
            plt.suptitle('Feature Distributions: Normal vs Outliers')
            plt.tight_layout()
            #plt.show()
            plt.savefig('outlier_analysis.png') # Save figure to file for ssh remote access
    
    # 5. PROBABILITY RANGE OUTLIERS
    print("\n=== PROBABILITY RANGE ANALYSIS ===")
    extreme_low = (y_true < 0.1) & outlier_mask
    extreme_high = (y_true > 0.9) & outlier_mask  
    mid_range = ((y_true >= 0.1) & (y_true <= 0.9)) & outlier_mask
    
    print(f"Outliers in extreme low range (0-0.1): {extreme_low.sum()}")
    print(f"Outliers in extreme high range (0.9-1.0): {extreme_high.sum()}")
    print(f"Outliers in mid range (0.1-0.9): {mid_range.sum()}")
    
    # 6. DATA QUALITY CHECKS
    print("\n=== DATA QUALITY CHECKS ===")
    if X_features is not None:
        # Check for NaN, inf, extreme values
        nan_count = np.isnan(X_features).sum()
        inf_count = np.isinf(X_features).sum()
        
        print(f"NaN values in features: {nan_count}")
        print(f"Inf values in features: {inf_count}")

        
        print("Number of feature names:", len(feature_names))
        print("Number of features in data:", X_features.shape[1])

        
 
        # Feature value ranges
        feature_stats = pd.DataFrame({
            'Feature': feature_names,
            'Min': X_features.min(axis=0),
            'Max': X_features.max(axis=0),
            'Mean': X_features.mean(axis=0),
            'Std': X_features.std(axis=0)
        })
        
        # Identify features with extreme ranges
        extreme_features = feature_stats[
            (feature_stats['Max'] - feature_stats['Min']) > 1000
        ]
        
        if len(extreme_features) > 0:
            print(f"\nFeatures with extreme ranges (>1000):")
            print(extreme_features[['Feature', 'Min', 'Max']])
    
    # 7. RETURN OUTLIER INFO
    outlier_info = {
        'indices': outlier_indices,
        'residuals': residuals[outlier_mask],
        'true_values': y_true[outlier_mask],
        'pred_values': y_pred[outlier_mask],
        'threshold': outlier_threshold
    }
    
    if X_features is not None:
        outlier_info['features'] = X_features[outlier_mask]
        outlier_info['significant_features'] = outlier_features
    
    return outlier_info

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
            
            # Collect node features (aggregate per graph for graph-level prediction)
            batch_size = data.batch.max().item() + 1
            for i in range(batch_size):
                mask = data.batch == i
                # Use mean node features per graph as graph features
                graph_features = data.x[mask].mean(dim=0)
                all_features.append(graph_features.cpu())
    
    # Convert to numpy
    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_trues).numpy()
    features = torch.stack(all_features).numpy()
    
    # For root_probability mode, take probability of class 1
    if mode == "root_probability" and preds.ndim == 1:
        pass  # Already single probability values
    elif mode == "distribution":
        trues = trues[:, 1]  # Take class 1 probability
        preds = preds[:, 1]
    
    # Feature names for your GNN features
    feature_names = [
        'node_type',        # 0
        'in_degree',        # 1
        'out_degree',       # 2
        'betweenness',      # 3
        'closeness',        # 4
        'pagerank',         # 5
        'degree_centrality',# 6
        'variable_card',    # 7
        'num_parents',      # 8
        'evidence_flag',    # 9
        'cpd_0',            # 10
        'cpd_1',            # 11
        'cpd_2',            # 12
        'cpd_3',            # 13
        'cpd_4',            # 14
        'cpd_5',            # 15
        'cpd_6',            # 16  <-- new CPD columns
        'cpd_7'             # 17
    ]

    
    # Run outlier analysis
    print("=== RUNNING OUTLIER ANALYSIS ===")
    outlier_info = analyze_outliers(
        y_true=trues,
        y_pred=preds,
        X_features=features,
        feature_names=feature_names,
        outlier_percentile=95
    )
    
    return outlier_info

# EXAMPLE USAGE IN YOUR TRAINING SCRIPT:
"""
# After training, add this to your main code:
if params['enable_outlier_analysis']:
    outlier_info = run_outlier_analysis_for_gnn(
        model=model,
        test_loader=test_loader, 
        device=device,
        mode=mode
    )
    
    # Print worst cases
    print(f"\\nWorst 5 outliers:")
    for i in range(min(5, len(outlier_info['indices']))):
        idx = outlier_info['indices'][i]
        print(f"Sample {idx}: True={outlier_info['true_values'][i]:.3f}, "
              f"Pred={outlier_info['pred_values'][i]:.3f}, "
              f"Error={outlier_info['residuals'][i]:.3f}")
"""