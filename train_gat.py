import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
import time
import json
from early_stopping_pytorch import EarlyStopping
from gat_model import GAT
import wandb
from outlier_analysis import analyze_outliers, run_outlier_analysis_for_gnn, analyze_graph_structure_outliers
from temperature_scaling import TemperatureScaling
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score

# --------------------------------------------
# Load configuration
# --------------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --------------------------------------------
# Create output directory for all results
# --------------------------------------------
OUTPUT_DIR = config.get("output_directory", "training_results_gat")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)

# --------------------------------------------
# Device setup and training parameters
# --------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = config["mode"]  # distribution / root_probability / regression
mask_strategy = config.get("mask_strategy", "root_only")  # root_only / both

# Hyperparameters (with defaults)
params = {
    'lr': float(config.get("learning_rate", 0.001)),
    'hidden_channels': int(config.get("hidden_channels", 32)),
    'dropout': float(config.get("dropout", 0.5)),
    'batch_size': int(config.get("batch_size", 32)),
    'weight_decay': float(config.get("weight_decay", 5e-4)),
    'patience': int(config.get("patience", 5)),
    'epochs': int(config.get("epochs", 100)),
    'heads': int(config.get("heads", 8)),
    'debug_attention': config.get("debug_attention", False),
    'enable_outlier_analysis': config.get("enable_outlier_analysis", True),
}

# --------------------------------------------
# Optional: initialize WandB
# --------------------------------------------
# wandb.init(project="gat_hyperparam_tuning", config=params)
# params = wandb.config

# ============================================
# Load datasets based on config (K-Fold or Classic)
# ============================================

def load_datasets_from_config(config):
    """
    Load datasets based on config settings
    - If use_kfold=true: loads all folds
    - If use_kfold=false: loads classic train/val/test split
    
    Returns:
        If K-fold: (list of tuples, metadata, True)
        If classic: ((train, val, test), None, False)
    """
    
    use_kfold = config.get('use_kfold', False)
    
    if use_kfold:
        # ==========================================
        # K-FOLD MODE: Load all folds
        # ==========================================
        print("="*60)
        print("K-FOLD MODE ENABLED")
        print("="*60)
        
        fold_metadata_path = "datasets/folds/fold_metadata.json"
        
        if not os.path.exists(fold_metadata_path):
            raise FileNotFoundError(
                "K-fold enabled in config but fold data not found!\n"
                "Please run data preprocessing first with use_kfold=true"
            )
        
        # Load metadata
        with open(fold_metadata_path, "r") as f:
            fold_metadata = json.load(f)
        
        n_folds = fold_metadata["k_folds"]
        print(f"Found {n_folds} folds in metadata")
        print(f"Stratified: {fold_metadata.get('stratified', 'N/A')}")
        print(f"Use log prob: {fold_metadata.get('use_log_prob', 'N/A')}")
        
        # Load all folds
        all_folds = []
        for fold_idx in range(n_folds):
            train_path = f"datasets/folds/fold_{fold_idx}_train.pt"
            val_path = f"datasets/folds/fold_{fold_idx}_val.pt"
            test_path = f"datasets/folds/fold_{fold_idx}_test.pt"
            
            if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
                raise FileNotFoundError(f"❌ Fold {fold_idx} files incomplete!")
            
            train_set = torch.load(train_path, weights_only=False)
            val_set = torch.load(val_path, weights_only=False)
            test_set = torch.load(test_path, weights_only=False)
            
            all_folds.append((train_set, val_set, test_set))
            
            print(f"  ✓ Fold {fold_idx}: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        
        print(f"✓ Successfully loaded all {n_folds} folds")
        print("="*60 + "\n")
        
        return all_folds, fold_metadata, True
    
    else:
        # ==========================================
        # CLASSIC MODE: Load single split
        # ==========================================
        print("="*60)
        print("CLASSIC SPLIT MODE")
        print("="*60)
        
        train_path = "datasets/train.pt"
        val_path = "datasets/val.pt"
        test_path = "datasets/test.pt"
        
        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            raise FileNotFoundError(
                "Classic split files not found!\n"
                "Please run data preprocessing first with use_kfold=false"
            )
        
        train_set = torch.load(train_path, weights_only=False)
        val_set = torch.load(val_path, weights_only=False)
        test_set = torch.load(test_path, weights_only=False)
        
        print(f"✓ Train: {len(train_set)}")
        print(f"✓ Val:   {len(val_set)}")
        print(f"✓ Test:  {len(test_set)}")
        print("="*60 + "\n")
        
        return (train_set, val_set, test_set), None, False


# ============================================
# LOAD DATA
# ============================================
try:
    data, fold_metadata, is_kfold = load_datasets_from_config(config)
except Exception as e:
    raise RuntimeError(f"Error loading datasets: {e}")

# ============================================
# DEBUG INFO
# ============================================
if is_kfold:
    # Debug first fold
    train_set, val_set, test_set = data[0]
    print("=== DEBUG INFO (Fold 0) ===")
else:
    train_set, val_set, test_set = data
    print("=== DEBUG INFO ===")

sample_data = train_set[0]
print(f"Sample data keys: {sample_data.keys()}")
print(f"Sample x shape: {sample_data.x.shape}")
print(f"Sample y shape: {sample_data.y.shape}")
print(f"Sample y type: {type(sample_data.y)}")
print(f"Sample y: {sample_data.y}")
print(f"Mode: {mode}")
print(f"Mask strategy: {mask_strategy}")
print(f"Device: {device}")
print(f"Debug attention: {params['debug_attention']}")
print("="*60 + "\n")

# ============================================
# DETERMINE TRAINING MODE
# ============================================
if is_kfold:
    print("Will perform K-FOLD CROSS-VALIDATION")
    print(f"   Number of folds: {len(data)}")
    print(f"   Training mode: Cross-validation")
else:
    print("Will perform SINGLE SPLIT TRAINING")
    print(f"   Training mode: Standard train/val/test")

print("\n" + "="*60 + "\n")


# --------------------------------------------
# Load global CPD length for regression
# --------------------------------------------
with open(config["global_cpd_len_path"], "r") as f:
    global_cpd_len = int(f.read())

in_channels = train_set[0].x.shape[1]

# --------------------------------------------
# Get log-probability flag from config
# --------------------------------------------
use_log_prob = config.get("use_log_prob", False)
print(f"Using log-probability mode: {use_log_prob}")

# --------------------------------------------
# Loss functions (updated for log-probability support)
# --------------------------------------------
def asymmetric_weighted_loss(preds, targets, 
                             underprediction_penalty=5.0, 
                             low_target_penalty=3.0, 
                             use_log_prob=False):
    """
    A loss function that heavily penalizes underpredictions and predictions
    associated with low target values in safety-critical scenarios.
    Auto-adjusts thresholds based on log-probability mode.
    """
    # Auto-adjust threshold based on log-prob mode
    low_target_threshold = -2.303 if use_log_prob else 0.1  # log(0.1) = -2.303
    
    diff = targets - preds
    squared_error = diff ** 2
    weights = torch.ones_like(targets, dtype=torch.float32)
    
    # Apply Underprediction Penalty
    underprediction_mask = diff > 0
    weights[underprediction_mask] *= underprediction_penalty
    
    # Apply Low Target Value Penalty
    low_target_mask = targets < low_target_threshold
    weights[low_target_mask] *= low_target_penalty
    
    weighted_loss = weights * squared_error
    return weighted_loss.mean()

def quantile_loss(preds, targets, tau=0.8):
    """Quantile loss - works the same in both probability and log-probability space"""
    diff = targets - preds
    return torch.mean(torch.max(tau * diff, (tau - 1) * diff))

def smart_quantile_loss(preds, targets, tau=0.8):
    """Smart quantile loss - works the same in both probability and log-probability space"""
    diff = targets - preds
    underestimate_mask = diff > 0
    normal_loss = F.mse_loss(preds[~underestimate_mask], targets[~underestimate_mask], reduction='none')
    quant_loss = torch.max(tau * diff[underestimate_mask], (tau - 1) * diff[underestimate_mask])
    return torch.cat([normal_loss, quant_loss]).mean()

def simple_asymmetric_loss(preds, targets, penalty=2.0):
    """Simple asymmetric loss - works the same in both probability and log-probability space"""
    diff = targets - preds
    loss = torch.where(diff > 0, penalty * diff**2, diff**2)
    return loss.mean()
    
def weighted_mse_loss(pred, target, use_log_prob=False):
    """
    Weighted MSE loss with support for log-probability space.
    """
    weights = torch.ones_like(target)
    
    if use_log_prob:
        # Log-probability thresholds
        weights[target < -2.303] = 5.0  # log(0.1) - low probability
        weights[target > -0.105] = 2.0  # log(0.9) - high probability
    else:
        # Raw probability thresholds
        weights[target < 0.1] = 5.0
        weights[target > 0.9] = 2.0
    
    return (weights * (pred - target) ** 2).mean()

def high_risk_focused_loss(preds, targets, 
                           base_penalty=2.0,
                           high_risk_penalty=10.0,
                           use_log_prob=False):
    """
    Smart loss: aggressive only on high-risk underpredictions.
    Auto-adjusts thresholds based on log-probability mode.
    """
    # Auto-adjust threshold based on log-prob mode
    high_risk_threshold = -0.357 if use_log_prob else 0.7  # log(0.7) = -0.357
    
    diff = targets - preds
    squared_error = diff ** 2
    
    # Default weights
    weights = torch.ones_like(targets)
    
    # Identify high-risk cases
    high_risk = targets > high_risk_threshold
    underpredict = diff > 0
    
    # Apply penalties
    weights[underpredict] *= base_penalty              # 2x for any underpredict
    weights[high_risk & underpredict] *= high_risk_penalty  # 10x for high-risk underpredict
    
    return (weights * squared_error).mean()

# --------------------------------------------
# Determine output channels and active loss
# --------------------------------------------
if mode == "distribution":
    out_channels = 2
    loss_fn = lambda pred, true: F.kl_div(F.log_softmax(pred, dim=1), true, reduction='batchmean')
    
elif mode == "root_probability":
    out_channels = 1
    
    # Use asymmetric weighted loss with log-prob support
    loss_fn = lambda pred, true: asymmetric_weighted_loss(
        pred, true, 
        underprediction_penalty=7.0, 
        low_target_penalty=3.0, 
        use_log_prob=use_log_prob  # Pass the flag from config
    )
    
    # Alternative: high_risk_focused_loss (uncomment to use)
    # loss_fn = lambda pred, true: high_risk_focused_loss(
    #     pred, true,
    #     base_penalty=1.5,
    #     high_risk_penalty=5.0,
    #     use_log_prob=use_log_prob  # Pass the flag from config
    # )
    
else:
    out_channels = global_cpd_len
    loss_fn = torch.nn.MSELoss()

print(f"Model config - Input: {in_channels}, Output: {out_channels}")
if mode == "root_probability":
    print(f"Loss function mode: {'Log-Probability' if use_log_prob else 'Raw Probability'}")

# --------------------------------------------
# Initialize model and optimizer
# --------------------------------------------
def initialize_model():
    model = GAT(
        in_channels=in_channels,
        hidden_channels=params['hidden_channels'],
        out_channels=out_channels,
        dropout=params['dropout'],
        heads=params['heads'],
        mode=mode,
        use_log_prob=use_log_prob
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    early_stopping = EarlyStopping(patience=params['patience'], verbose=True)
    return model, optimizer, early_stopping

# --------------------------------------------
# Helper: debug batch
# --------------------------------------------
def debug_batch(data, batch_idx=0):
    print(f"\n=== BATCH DEBUG {batch_idx} ===")
    print(f"Batch size: {data.batch.max().item() + 1}")
    print(f"Total nodes in batch: {data.x.shape[0]}")
    print(f"Data y shape: {data.y.shape}")
    print(f"Data y: {data.y}")
    print(f"Unique batch indices: {torch.unique(data.batch)}")
    print(f"Root nodes: {data.root_node if hasattr(data, 'root_node') else 'Not available'}")

# --------------------------------------------
# Helper: extract targets and predictions
# --------------------------------------------
def extract_targets_and_predictions(data, out):
    batch_size = data.batch.max().item() + 1
    if mode == "distribution":
        targets = data.y.view(batch_size, 2)
        targets = targets / targets.sum(dim=1, keepdim=True)
        predictions = out
    elif mode == "root_probability":
        targets = data.y.squeeze()
        predictions = out.squeeze()
        assert targets.shape == predictions.shape
    else:
        targets = data.y
        predictions = out
    return targets, predictions

# --------------------------------------------
# Training function for one epoch
# --------------------------------------------
def train_epoch(model, optimizer, loader):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if batch_idx == 0:
            debug_batch(data, batch_idx)
        
        optimizer.zero_grad()
        
        # GAT returns tuple if debug_attention=True
        if params['debug_attention'] and batch_idx == 0:
            out, attn_dict = model(data, debug_attention=True)
            print(f"\n=== Attention Weights (Batch {batch_idx}) ===")
            for layer_name, attn in attn_dict.items():
                print(f"{layer_name}: shape={attn.shape}")
        else:
            out = model(data, debug_attention=False)
        
        targets, predictions = extract_targets_and_predictions(data, out)
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        # Check prediction ranges
        if batch_idx == 0:
            print(f"\n=== Prediction Check (Batch {batch_idx}) ===")
            print(f"Loss: {loss.item():.4f}")
            print(f"Pred range: [{predictions.min():.3f}, {predictions.max():.3f}]")
            print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# --------------------------------------------
# Evaluation function with enhanced metrics
# --------------------------------------------
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_preds, all_true = [], []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            
            # GAT returns tuple if debug_attention=True
            if params['debug_attention'] and batch_idx == 0:
                out, attn_dict = model(data, debug_attention=True)
                print(f"\n=== Attention Weights During Eval (Batch {batch_idx}) ===")
                for layer_name, attn in attn_dict.items():
                    print(f"{layer_name}: shape={attn.shape}")
            else:
                out = model(data, debug_attention=False)
            
            targets, predictions = extract_targets_and_predictions(data, out)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_true.append(targets.cpu())
    
    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_true).numpy()
    metrics = {"loss": total_loss / len(loader)}
    
    # Debug: Check value ranges for root_probability mode
    if mode == "root_probability":
        print(f"\n=== DEBUG EVALUATE ===")
        print(f"use_log_prob: {use_log_prob}")
        print(f"Raw preds range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"Raw trues range: [{trues.min():.4f}, {trues.max():.4f}]")
        print(f"Raw preds sample (first 5): {preds[:5]}")
        print(f"Raw trues sample (first 5): {trues[:5]}")
    
    if mode == "distribution":
        softmax_preds = F.softmax(torch.tensor(preds), dim=1).numpy()
        pred_labels = np.argmax(softmax_preds, axis=1)
        true_labels = np.argmax(trues, axis=1)
        metrics["accuracy"] = np.mean(pred_labels == true_labels)
        
    elif mode == "root_probability":
        # Compute metrics based on log-prob or raw prob
        if use_log_prob:
            # Metrics in log-space for training monitoring
            metrics["mae"] = np.mean(np.abs(preds - trues))
            metrics["rmse"] = np.sqrt(np.mean((preds - trues)**2))
            metrics["mse"] = np.mean((preds - trues)**2)
            metrics["r2_score"] = r2_score(trues, preds)
            
            # For interpretable metrics, convert back carefully
            # Clamp to avoid numerical issues: log(prob) ∈ [-∞, 0]
            preds_clamped = np.clip(preds, -10, 0)  # Avoid exp(-∞) = 0
            trues_clamped = np.clip(trues, -10, 0)
            
            preds_prob = np.exp(preds_clamped)
            trues_prob = np.exp(trues_clamped)
            
            print(f"After exp preds range: [{preds_prob.min():.4f}, {preds_prob.max():.4f}]")
            print(f"After exp trues range: [{trues_prob.min():.4f}, {trues_prob.max():.4f}]")
            print(f"Preds_prob sample (first 5): {preds_prob[:5]}")
            print(f"Trues_prob sample (first 5): {trues_prob[:5]}")
            
        else:
            # Raw probability space
            preds_prob = preds
            trues_prob = trues
            
            # Standard metrics
            metrics["mae"] = np.mean(np.abs(preds_prob - trues_prob))
            metrics["rmse"] = np.sqrt(np.mean((preds_prob - trues_prob)**2))
            metrics["mse"] = np.mean((preds_prob - trues_prob)**2)
            metrics["r2_score"] = r2_score(trues_prob, preds_prob)
        
        # Tolerance-based accuracy (in probability space)
        tolerances = [0.05, 0.10, 0.15]
        for tol in tolerances:
            within_tolerance = np.abs(preds_prob - trues_prob) <= tol
            metrics[f"accuracy_within_{int(tol*100)}pct"] = np.mean(within_tolerance)
        
        # Asymmetric error analysis (in probability space)
        errors = trues_prob - preds_prob
        under_mask = errors > 0
        over_mask = errors < 0
        
        metrics["underpredict_rate"] = np.mean(under_mask)
        metrics["overpredict_rate"] = np.mean(over_mask)
        metrics["mean_underpredict_error"] = np.mean(errors[under_mask]) if np.any(under_mask) else 0.0
        metrics["mean_overpredict_error"] = np.mean(np.abs(errors[over_mask])) if np.any(over_mask) else 0.0
        
        # High-risk case analysis (targets > 0.7 in probability space)
        high_risk_mask = trues_prob > 0.7
        if np.any(high_risk_mask):
            metrics["high_risk_mae"] = np.mean(np.abs(preds_prob[high_risk_mask] - trues_prob[high_risk_mask]))
            metrics["high_risk_underpredict_rate"] = np.mean(preds_prob[high_risk_mask] < trues_prob[high_risk_mask])
        else:
            metrics["high_risk_mae"] = 0.0
            metrics["high_risk_underpredict_rate"] = 0.0
        
        # Low-risk case analysis (targets < 0.3)
        low_risk_mask = trues_prob < 0.3
        if np.any(low_risk_mask):
            metrics["low_risk_mae"] = np.mean(np.abs(preds_prob[low_risk_mask] - trues_prob[low_risk_mask]))
        else:
            metrics["low_risk_mae"] = 0.0
        
        # Percentile errors (in probability space)
        abs_errors = np.abs(preds_prob - trues_prob)
        metrics["p50_error"] = np.percentile(abs_errors, 50)
        metrics["p95_error"] = np.percentile(abs_errors, 95)
        metrics["p99_error"] = np.percentile(abs_errors, 99)
        
        # Calibration (in probability space)
        metrics["mean_prediction"] = np.mean(preds_prob)
        metrics["mean_ground_truth"] = np.mean(trues_prob)
        metrics["mean_bias"] = np.mean(preds_prob) - np.mean(trues_prob)
        
    else:  # regression
        metrics["mae"] = np.mean(np.abs(preds - trues))
        metrics["rmse"] = np.sqrt(np.mean((preds - trues)**2))
        metrics["r2_score"] = r2_score(trues, preds)
    
    return metrics

# --------------------------------------------
# Plotting functions
# --------------------------------------------
def plot_true_vs_predicted(loader, model, fold_idx=None):
    """Plot true vs predicted values for root_probability mode"""
    model.eval()
    all_preds, all_true = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data, debug_attention=False)
            targets, predictions = extract_targets_and_predictions(data, out)
            all_preds.append(predictions.cpu())
            all_true.append(targets.cpu())
    
    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_true).numpy()
    
    # Convert from log-prob to prob if needed
    if use_log_prob:
        preds_clamped = np.clip(preds, -10, 0)
        trues_clamped = np.clip(trues, -10, 0)
        preds = np.exp(preds_clamped)
        trues = np.exp(trues_clamped)
    
    # Ensure values are in [0, 1]
    preds = np.clip(preds, 0, 1)
    trues = np.clip(trues, 0, 1)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(trues, preds, alpha=0.5, s=20)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True Probability', fontsize=12)
    plt.ylabel('Predicted Probability', fontsize=12)
    plt.title('True vs Predicted Probabilities', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    
    filename = f'true_vs_pred{"_fold_" + str(fold_idx) if fold_idx is not None else ""}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", filename), dpi=300)
    plt.close()
    print(f"Saved true vs predicted plot to {os.path.join(OUTPUT_DIR, 'plots', filename)}")

# --------------------------------------------
# Full training loop with K-Fold + AGGREGATION
# --------------------------------------------
def run_training(train_set, val_set, test_set):
    if config.get("use_kfold", False):
        print(f"Running {config['k_folds']}-Fold Cross Validation...")
        full_dataset = train_set + val_set
        
        # Storage for aggregating results across folds
        all_fold_metrics = []
        
        if config.get("stratify_folds", False) and mode == "root_probability":
            # For log-prob mode, stratify based on whether log-prob > log(0.5) = -0.693
            if use_log_prob:
                stratify = torch.tensor([g.y.item() > -0.693 for g in full_dataset])
            else:
                stratify = torch.tensor([g.y.item() > 0.5 for g in full_dataset])
            skf = StratifiedKFold(n_splits=config['k_folds'], shuffle=True, random_state=config.get('fold_random_seed', 42))
            splits = skf.split(np.arange(len(full_dataset)), stratify)
        else:
            kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=config.get('fold_random_seed', 42))
            splits = kf.split(full_dataset)
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n{'='*60}")
            print(f"=== Fold {fold_idx+1}/{config['k_folds']} ===")
            print(f"{'='*60}")
            
            fold_train = [full_dataset[i] for i in train_idx]
            fold_val = [full_dataset[i] for i in val_idx]
            train_loader = DataLoader(fold_train, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(fold_val, batch_size=params['batch_size'])
            test_loader = DataLoader(test_set, batch_size=params['batch_size'])
            
            model, optimizer, early_stopping = initialize_model()
            fold_metrics = train_model(model, optimizer, early_stopping, train_loader, val_loader, test_loader, fold_idx=fold_idx)
            
            # Store fold metrics for aggregation
            all_fold_metrics.append(fold_metrics)
        
        # ============================================
        # AGGREGATE RESULTS ACROSS ALL FOLDS
        # ============================================
        print(f"\n{'='*80}")
        print(f"{'='*80}")
        print(f"AGGREGATED RESULTS ACROSS {config['k_folds']} FOLDS")
        print(f"{'='*80}")
        print(f"{'='*80}\n")
        
        # Compute mean and std for each metric
        aggregated_metrics = {}
        metric_keys = all_fold_metrics[0].keys()
        
        for key in metric_keys:
            values = [fold[key] for fold in all_fold_metrics]
            aggregated_metrics[f"{key}_mean"] = np.mean(values)
            aggregated_metrics[f"{key}_std"] = np.std(values)
        
        # Print aggregated results
        if mode == "root_probability":
            print(f"--- Core Regression Metrics ---")
            print(f"MAE: {aggregated_metrics['mae_mean']:.4f} ± {aggregated_metrics['mae_std']:.4f}")
            print(f"RMSE: {aggregated_metrics['rmse_mean']:.4f} ± {aggregated_metrics['rmse_std']:.4f}")
            print(f"R² Score: {aggregated_metrics['r2_score_mean']:.4f} ± {aggregated_metrics['r2_score_std']:.4f}")
            
            print(f"\n--- Tolerance-Based Accuracy ---")
            print(f"Within 5%: {aggregated_metrics['accuracy_within_5pct_mean']:.4f} ± {aggregated_metrics['accuracy_within_5pct_std']:.4f}")
            print(f"Within 10%: {aggregated_metrics['accuracy_within_10pct_mean']:.4f} ± {aggregated_metrics['accuracy_within_10pct_std']:.4f}")
            print(f"Within 15%: {aggregated_metrics['accuracy_within_15pct_mean']:.4f} ± {aggregated_metrics['accuracy_within_15pct_std']:.4f}")
            
            print(f"\n--- Safety-Critical Metrics ---")
            print(f"Underpredict Rate: {aggregated_metrics['underpredict_rate_mean']:.4f} ± {aggregated_metrics['underpredict_rate_std']:.4f}")
            print(f"High-Risk MAE: {aggregated_metrics['high_risk_mae_mean']:.4f} ± {aggregated_metrics['high_risk_mae_std']:.4f}")
            print(f"High-Risk Underpredict Rate: {aggregated_metrics['high_risk_underpredict_rate_mean']:.4f} ± {aggregated_metrics['high_risk_underpredict_rate_std']:.4f}")
            
            print(f"\n--- Error Distribution ---")
            print(f"P95 Error: {aggregated_metrics['p95_error_mean']:.4f} ± {aggregated_metrics['p95_error_std']:.4f}")
            print(f"P99 Error: {aggregated_metrics['p99_error_mean']:.4f} ± {aggregated_metrics['p99_error_std']:.4f}")
            
            print(f"\n--- Calibration ---")
            print(f"Mean Bias: {aggregated_metrics['mean_bias_mean']:.4f} ± {aggregated_metrics['mean_bias_std']:.4f}")
        
        # Save aggregated metrics to file
        aggregated_filename = "aggregated_kfold_metrics.txt"
        with open(os.path.join(OUTPUT_DIR, "metrics", aggregated_filename), "w") as f:
            f.write(f"K-Fold Cross-Validation Results ({config['k_folds']} folds)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Log-probability: {use_log_prob}\n")
            f.write("="*80 + "\n\n")
            f.write("Format: metric_mean ± metric_std\n")
            f.write("="*80 + "\n\n")
            for key, value in aggregated_metrics.items():
                f.write(f"{key}: {value:.6f}\n")
        
        print(f"\n{'-'*80}")
        print(f"Aggregated metrics saved to {os.path.join(OUTPUT_DIR, 'metrics', aggregated_filename)}")
        print(f"{'-'*80}\n")
        
    else:
        # Single train/val/test split (no K-Fold)
        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=params['batch_size'])
        test_loader = DataLoader(test_set, batch_size=params['batch_size'])
        
        model, optimizer, early_stopping = initialize_model()
        train_model(model, optimizer, early_stopping, train_loader, val_loader, test_loader)

# --------------------------------------------
# Modified train_model to RETURN metrics
# --------------------------------------------
def train_model(model, optimizer, early_stopping, train_loader, val_loader, test_loader, fold_idx=None):
    train_losses, val_metrics, epoch_times = [], [], []
    fold_suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
    
    print(f"\nStarting training{' for fold ' + str(fold_idx) if fold_idx is not None else ''}...")

    for epoch in range(1, params['epochs'] + 1):
        start_time = time.time()
        train_loss = train_epoch(model, optimizer, train_loader)
        metrics = evaluate(model, val_loader)
        val_metrics.append(metrics)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        train_losses.append(train_loss)

        # Enhanced printing for root_probability mode
        if mode == "root_probability":
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {metrics.get('loss', float('nan')):.4f} | "
                  f"MAE: {metrics.get('mae', 0):.4f} | "
                  f"RMSE: {metrics.get('rmse', 0):.4f} | "
                  f"R²: {metrics.get('r2_score', 0):.4f} | "
                  f"Under Rate: {metrics.get('underpredict_rate', 0):.3f} | "
                  f"Time: {epoch_time:.2f}s")
        else:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {metrics.get('loss', float('nan')):.4f} | Time: {epoch_time:.2f}s")

        early_stopping(metrics["loss"], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Temperature scaling (if enabled)
    if config.get("use_temperature_scaling", False):
        temp_scaler = TemperatureScaling(init_temp=1.0)
        calibrated_temp = temp_scaler.calibrate(model=model, loader=val_loader, device=device, mode=config["mode"])
        print(f"Learned temperature: {calibrated_temp:.4f}")

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print(f"Final Test Results (Fold {fold_idx if fold_idx is not None else 'Single Run'}):")
    print(f"{'='*60}")
    test_metrics = evaluate(model, test_loader)
    
    # Print detailed metrics for root_probability mode
    if mode == "root_probability":
        print(f"\n--- Core Regression Metrics ---")
        print(f"MAE: {test_metrics.get('mae', 0):.4f}")
        print(f"RMSE: {test_metrics.get('rmse', 0):.4f}")
        print(f"MSE: {test_metrics.get('mse', 0):.4f}")
        print(f"R² Score: {test_metrics.get('r2_score', 0):.4f}")
        
        print(f"\n--- Tolerance-Based Accuracy ---")
        print(f"Within 5%: {test_metrics.get('accuracy_within_5pct', 0):.4f}")
        print(f"Within 10%: {test_metrics.get('accuracy_within_10pct', 0):.4f}")
        print(f"Within 15%: {test_metrics.get('accuracy_within_15pct', 0):.4f}")
        
        print(f"\n--- Safety-Critical Metrics ---")
        print(f"Underpredict Rate: {test_metrics.get('underpredict_rate', 0):.4f}")
        print(f"Mean Underpredict Error: {test_metrics.get('mean_underpredict_error', 0):.4f}")
        print(f"High-Risk MAE: {test_metrics.get('high_risk_mae', 0):.4f}")
        print(f"High-Risk Underpredict Rate: {test_metrics.get('high_risk_underpredict_rate', 0):.4f}")
        print(f"Low-Risk MAE: {test_metrics.get('low_risk_mae', 0):.4f}")
        
        print(f"\n--- Error Distribution ---")
        print(f"P50 Error (Median): {test_metrics.get('p50_error', 0):.4f}")
        print(f"P95 Error: {test_metrics.get('p95_error', 0):.4f}")
        print(f"P99 Error: {test_metrics.get('p99_error', 0):.4f}")
        
        print(f"\n--- Calibration ---")
        print(f"Mean Prediction: {test_metrics.get('mean_prediction', 0):.4f}")
        print(f"Mean Ground Truth: {test_metrics.get('mean_ground_truth', 0):.4f}")
        print(f"Mean Bias: {test_metrics.get('mean_bias', 0):.4f}")
        
        print(f"\n--- Timing ---")
        print(f"Avg Time/Epoch: {np.mean(epoch_times):.2f}s")
        print(f"Total Training Time: {np.sum(epoch_times):.2f}s")
    else:
        print(test_metrics)
    
    # Save per-fold metrics
    metrics_filename = f"test_metrics{fold_suffix}.txt"
    with open(os.path.join(OUTPUT_DIR, "metrics", metrics_filename), "w") as f:
        f.write(f"Test Metrics{' - Fold ' + str(fold_idx) if fold_idx is not None else ''}:\n")
        f.write("="*60 + "\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Log-probability: {use_log_prob}\n")
        f.write("="*60 + "\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"\nMetrics saved to {os.path.join(OUTPUT_DIR, 'metrics', metrics_filename)}")

    # Plot true vs predicted
    if mode == "root_probability":
        plot_true_vs_predicted(test_loader, model, fold_idx)

    # Outlier analysis
    if params['enable_outlier_analysis'] and mode == "root_probability":
        print(f"\n{'='*60}")
        print("Running Outlier Analysis...")
        print(f"{'='*60}")
        
        outlier_info = run_outlier_analysis_for_gnn(model, test_loader, device, mode)
        print(f"\nWorst 10 outliers for fold {fold_idx if fold_idx is not None else 'single run'}:")
        for i in range(min(10, len(outlier_info['indices']))):
            idx = outlier_info['indices'][i]
            print(f"Sample {idx}: True={outlier_info['true_values'][i]:.4f}, "
                  f"Pred={outlier_info['pred_values'][i]:.4f}, "
                  f"Error={outlier_info['residuals'][i]:.4f}")
        
        # Structural analysis (if available)
        try:
            structural_analysis = analyze_graph_structure_outliers(model, test_loader, device, mode)
        except Exception as e:
            print(f"Structural analysis skipped: {e}")

    # Save model
    evidence_type = "intermediate" if config.get("use_intermediate") else "leaf"
    log_suffix = "_logprob" if use_log_prob else ""
    model_filename = f"gat_{mode}_{mask_strategy}_{evidence_type}{log_suffix}{fold_suffix}.pt"
    model_path = os.path.join(OUTPUT_DIR, "models", model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Plot training curves
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot([m["loss"] for m in val_metrics], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.title("Training Curve", fontsize=14)
    plt.grid(True, alpha=0.3)

    if mode == "root_probability":
        plt.subplot(132)
        plt.plot([m.get("mae", 0) for m in val_metrics], label="MAE", linewidth=2)
        plt.plot([m.get("rmse", 0) for m in val_metrics], label="RMSE", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Error", fontsize=12)
        plt.legend(fontsize=10)
        plt.title("Validation Error Metrics", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(133)
        ax1 = plt.gca()
        ax1.plot([m.get("underpredict_rate", 0) for m in val_metrics], label="Underpredict Rate", color='red', linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Underpredict Rate", fontsize=12, color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Target (20%)')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        ax2.plot([m.get("r2_score", 0) for m in val_metrics], label="R² Score", color='blue', linewidth=2)
        ax2.set_ylabel("R² Score", fontsize=12, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.legend(loc='upper right', fontsize=10)
        plt.title("Safety & Quality Metrics", fontsize=14)
    
    elif mode == "distribution":
        plt.subplot(132)
        plt.plot([m.get("calibration_error", 0) for m in val_metrics], 
                 label="Calibration Error", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Calibration Error", fontsize=12)
        plt.legend(fontsize=10)
        plt.title("Calibration Error", fontsize=14)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = f'training_curve{fold_suffix}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", plot_filename), dpi=300)
    plt.close()
    print(f"Training curves saved to {os.path.join(OUTPUT_DIR, 'plots', plot_filename)}")
    
    # RETURN test metrics for aggregation
    return test_metrics

# --------------------------------------------
# Entry point
# --------------------------------------------
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Starting GAT Training")
    print(f"Mode: {mode}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Attention Heads: {params['heads']}")
    print(f"Debug Attention: {params['debug_attention']}")
    print(f"{'='*60}\n")
    
    run_training(train_set, val_set, test_set)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print(f"  - Models: {os.path.join(OUTPUT_DIR, 'models')}")
    print(f"  - Plots: {os.path.join(OUTPUT_DIR, 'plots')}")
    print(f"  - Metrics: {os.path.join(OUTPUT_DIR, 'metrics')}")
    print(f"{'='*60}\n")
    
    # wandb.finish()