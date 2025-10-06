# Train_graphsage.py


import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
import time
#from sklearn.metrics import confusion_matrix, roc_auc_score
from early_stopping_pytorch import EarlyStopping
from graphsage_model import GraphSAGE
import wandb
from outlier_analysis import analyze_outliers, run_outlier_analysis_for_gnn, analyze_graph_structure_outliers
from temperature_scaling import TemperatureScaling

   
# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set up device and training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = config["mode"]  # distribution/root_probability/regression
mask_strategy = config.get("mask_strategy", "root_only")  # root_only/both

# Hyperparameters from config with defaults
params = {
    'lr': float(config.get("learning_rate", 0.001)),
    'hidden_channels': int(config.get("hidden_channels", 32)),
    'dropout': float(config.get("dropout", 0.5)),
    'batch_size': int(config.get("batch_size", 32)),
    'weight_decay': float(config.get("weight_decay", 5e-4)),
    'patience': int(config.get("patience", 5)),
    'epochs': int(config.get("epochs", 100)),
    'enable_outlier_analysis': config.get("enable_outlier_analysis", True)
}

# Initialize wandb
# wandb.init(
#     project="graphsage_hyperparam_tuning",
#     config={
#         "lr": 0.001,
#         "hidden_channels": 32,
#         "dropout": 0.5,
#         "batch_size": 32,
#         "weight_decay": 5e-4,
#         "patience": 5,
#         "epochs": 100
#     }
# )
# params = wandb.config


# Load datasets
try:
    train_set = torch.load("datasets/train.pt", weights_only=False)
    val_set = torch.load("datasets/val.pt", weights_only=False)
    test_set = torch.load("datasets/test.pt", weights_only=False)
    print(f"Loaded datasets - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
except Exception as e:
    raise RuntimeError(f"Error loading datasets: {e}")

# Debug: Check data structure
print("=== DEBUG INFO ===")
sample_data = train_set[0]
print(f"Sample data keys: {sample_data.keys()}")
print(f"Sample x shape: {sample_data.x.shape}")
print(f"Sample y shape: {sample_data.y.shape}")
print(f"Sample y type: {type(sample_data.y)}")
print(f"Sample y: {sample_data.y}")
print(f"Mode: {mode}")

# Check if y dimensions are consistent
y_shapes = [data.y.shape for data in train_set[:10]]
print(f"First 10 y shapes: {y_shapes}")

# Initialize data loaders
train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=params['batch_size'])
test_loader = DataLoader(test_set, batch_size=params['batch_size'])


# --- OVERFIT TEST BLOCK: only use 10 samples ---
# train_set = train_set[:10]
# val_set = val_set[:10]


# Debug: Check class imbalance
print("\n=== CLASS IMBALANCE DEBUG ===")
ys = torch.cat([data.y for data in train_set])
print("Class 0:", (ys < 0.5).sum().item())
print("Class 1:", (ys >= 0.5).sum().item())


# Load global CPD length
with open(config["global_cpd_len_path"], "r") as f:
    global_cpd_len = int(f.read())

in_channels = train_set[0].x.shape[1]

def quantile_loss(preds, targets, tau=0.8):
    """Quantile loss (also called pinball loss). tau ∈ (0, 1)"""
    diff = targets - preds
    return torch.mean(torch.max(tau * diff, (tau - 1) * diff))

def smart_quantile_loss(preds, targets, tau=0.8):
    """Only apply quantile loss where we're actually underestimating"""
    diff = targets - preds
    underestimate_mask = diff > 0  # Only where we predict too low
    
    # Normal MSE for good predictions, quantile for underestimates
    normal_loss = F.mse_loss(preds[~underestimate_mask], targets[~underestimate_mask], reduction='none')
    quantile_loss = torch.max(tau * diff[underestimate_mask], (tau - 1) * diff[underestimate_mask])
    
    return torch.cat([normal_loss, quantile_loss]).mean()

def simple_asymmetric_loss(preds, targets, penalty=2.0):
    """Simple asymmetric loss that penalizes underprediction more"""
    diff = targets - preds  # positive = underprediction, negative = overprediction
    
    # Apply different penalties
    loss = torch.where(diff > 0, 
                      penalty * diff**2,  # Heavy penalty for underprediction
                      diff**2)            # Normal penalty for overprediction
    return loss.mean()

# Determine model output size and loss fun ction
if mode == "distribution":
    out_channels = 2
    loss_fn = lambda pred, true: F.kl_div(F.log_softmax(pred, dim=1), true, reduction='batchmean')
    #loss_fn = torch.nn.CrossEntropyLoss() 
elif mode == "root_probability":
    out_channels = 1  # Single output
    #pos_weight = torch.tensor([894/106])  # Fix imbalance
    #loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    #loss_fn = torch.nn.BCEWithLogitsLoss()
    #loss_fn = torch.nn.MSELoss()
    #loss_fn = torch.nn.SmoothL1Loss()
    loss_fn = lambda pred, true: simple_asymmetric_loss(pred, true, penalty=1.8)
    #tau = float(config.get("quantile_tau", 0.75))  
    #loss_fn = lambda pred, true: smart_quantile_loss(pred, true, tau=tau)
    #loss_fn = lambda pred, true: quantile_loss(pred, true, tau=tau)
    
    
else:  # regression
    out_channels = global_cpd_len
    loss_fn = torch.nn.MSELoss()

print(f"Model config - Input: {in_channels}, Output: {out_channels}")

# Initialize model with proper device placement
model = GraphSAGE(
    in_channels=in_channels,
    hidden_channels=params['hidden_channels'],
    out_channels=out_channels,
    dropout=params['dropout']
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=params['lr'],
    weight_decay=params['weight_decay']
)
early_stopping = EarlyStopping(patience=params['patience'], verbose=True)

def debug_batch(data, batch_idx=0):
    """Debug function to check batch dimensions"""
    print(f"\n=== BATCH DEBUG {batch_idx} ===")
    print(f"Batch size: {data.batch.max().item() + 1}")
    print(f"Total nodes in batch: {data.x.shape[0]}")
    print(f"Data y shape: {data.y.shape}")
    print(f"Data y: {data.y}")
    print(f"Unique batch indices: {torch.unique(data.batch)}")
    print(f"Root nodes: {data.root_node if hasattr(data, 'root_node') else 'Not available'}")

def extract_targets_and_predictions(data, out, batch_idx=0):
    
    batch_size = data.batch.max().item() + 1
    
    if mode == "distribution":
        # Reshape to [batch_size, 2] 
        targets = data.y.view(batch_size, 2)
        targets = targets / targets.sum(dim=1, keepdim=True)
        #temperature = 2.0
        #targets = F.softmax(torch.log(targets + 1e-8) / temperature, dim=1)
        predictions = out
        
    elif mode == "root_probability":
        # Extract probability of class 1 (every second element starting from 1)
        #targets = data.y.view(batch_size, 2)[:, 1] 
        #predictions = out.squeeze(-1)
        #targets = data.y.view(-1)
        #predictions = out.view(-1)
        targets = data.y.squeeze()
        predictions = out.squeeze()
        assert targets.shape == predictions.shape, f"Target shape: {targets.shape}, Pred shape: {predictions.shape}"

       
    else:  # regression
        targets = data.y
        predictions = out
    
    return targets, predictions

def train():
    """Training loop with mode-specific handling"""
    model.train()
    total_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        
        # Debug first batch
        if batch_idx == 0:
            print("Input features sample:", data.x[:5])
            debug_batch(data, batch_idx)
        
        optimizer.zero_grad()
        out = model(data)
        
        # Debug model output
        if batch_idx == 0:
            print("Model output sample:", out[:5])
            print(f"Model output shape: {out.shape}")
            print(f"Model output: {out}")
            #print(f"Model output (sigmoid probabilities): {torch.sigmoid(out)}")
            print("Target y:", data.y[:10]) ###########
        
        # Extract targets and predictions properly
        targets, predictions = extract_targets_and_predictions(data, out, batch_idx)
        
        # Compute loss
        loss = loss_fn(predictions, targets)
        loss.backward()

        # Gradient check: print norm of gradients for conv layers
        if batch_idx == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Grad norm for {name}: {param.grad.norm().item():.6f}")


        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_true = []
    
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            targets, predictions = extract_targets_and_predictions(data, out)
            
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            
            # Handle scalar prediction case (0D tensor) by unsqueezing
            if predictions.dim() == 0:
                all_preds.append(predictions.cpu().unsqueeze(0))
            else:
                all_preds.append(predictions.cpu())
            
            if targets.dim() == 0:
                all_true.append(targets.cpu().unsqueeze(0))
            else:
                all_true.append(targets.cpu())

            # Old version (caused crash for 0D tensors)
            # all_preds.append(predictions.cpu())
            # all_true.append(targets.cpu())
    
    # Robust stack (can handle 1D or 2D safely)
    preds = torch.cat(all_preds, dim=0).numpy()
    trues = torch.cat(all_true, dim=0).numpy()
    metrics = {
        "loss": total_loss / len(loader),
    }
    
    # metrics = {
    #     "loss": total_loss / len(loader),
    #     "mae": np.mean(np.abs(preds - trues)),
    #     "rmse": np.sqrt(np.mean((preds - trues) ** 2))
    # }
    
    if mode == "distribution":
        softmax_preds = F.softmax(torch.tensor(preds), dim=1).numpy()
        pred_labels = np.argmax(softmax_preds, axis=1)
        true_labels = np.argmax(trues, axis=1)
        metrics["accuracy"] = np.mean(pred_labels == true_labels)
        
    elif mode == "root_probability":
        try:
            # Basic accuracy
            binary_preds = (preds > 0.5).astype(int)
            binary_trues = (trues > 0.5).astype(int)
            metrics["accuracy"] = np.mean(binary_preds == binary_trues)
            
            # Underprediction analysis
            diff = trues - preds
            underpredict_mask = diff > 0
            metrics["underpredict_rate"] = np.mean(underpredict_mask)
            
            if np.any(underpredict_mask):
                metrics["avg_underpredict_error"] = np.mean(diff[underpredict_mask])
            else:
                metrics["avg_underpredict_error"] = 0.0
                
            # Only add MAE/RMSE if you really need them
            metrics["mae"] = np.mean(np.abs(preds - trues))
            metrics["rmse"] = np.sqrt(np.mean((preds - trues) ** 2))
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Fallback to just basic metrics
            pass
    
    else:  # regression
        metrics["mae"] = np.mean(np.abs(preds - trues))
        metrics["rmse"] = np.sqrt(np.mean((preds - trues) ** 2))
    
    return metrics


def plot_results(loader):
    """Mode-specific plotting of results"""
    model.eval()
    
    if mode == "distribution":
        plot_distribution_results(loader)
    elif mode == "root_probability":
        plot_root_probability_results(loader)
    else:  # regression
        plot_regression_results(loader)

def plot_distribution_results(loader):
    """Plot calibration curve for distribution mode"""
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            targets, predictions = extract_targets_and_predictions(data, out)
            
            preds.append(F.softmax(predictions, dim=1))
            trues.append(targets)
    
    preds = torch.cat(preds).cpu().numpy()
    trues = torch.cat(trues).cpu().numpy()
    
    # Calibration plot
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(preds[:,1], bins)
    
    calib_vals = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.any():
            calib_vals.append(trues[mask,1].mean())
        else:
            calib_vals.append(0)
    
    plt.figure(figsize=(6,6))
    plt.plot(bin_centers, calib_vals, 'o-', label='Model')
    plt.plot([0,1], [0,1], 'k--', label='Perfect')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_root_probability_results(loader):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            target, prediction = extract_targets_and_predictions(data, out)

            if prediction.dim() == 0:
                preds.append(prediction.cpu().unsqueeze(0))
            else:
                preds.append(prediction.cpu())

            if target.dim() == 0:
                trues.append(target.cpu().unsqueeze(0))
            else:
                trues.append(target.cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(trues, preds, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('True Probability')
    plt.ylabel('Predicted Probability')
    plt.title('True vs Predicted Values')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig('True vs Predicted Values.png')


def plot_regression_results(loader):
    """Scatter plot for regression mode"""
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            targets, predictions = extract_targets_and_predictions(data, out)
            
            preds.append(predictions)
            trues.append(targets)
    
    preds = torch.cat(preds).cpu().numpy()
    trues = torch.cat(trues).cpu().numpy()
    
    plt.figure(figsize=(6,6))
    plt.scatter(trues, preds, alpha=0.3)
    plt.plot([trues.min(), trues.max()], [trues.min(), trues.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted")
    plt.grid(True)
    plt.show()

# Training loop with metrics tracking
train_losses = []
val_metrics = []
criterion = torch.nn.MSELoss()
print("Starting training...")

epoch_times = []
for epoch in range(1, params['epochs'] + 1):
    # Training phase
    try:
        train_loss = train()
        train_losses.append(train_loss)
        start_time = time.time()
        # Validation phase
       # metrics = evaluate(val_loader)
        metrics = evaluate(model, val_loader, criterion)
        val_metrics.append(metrics)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Print epoch statistics
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
            #   f"Val Loss: {metrics['loss']:.4f}", end="")
            #f"Val Loss: {metrics['loss']:.4f} | "
            f"Val Loss: {metrics.get('loss', float('nan')):.4f} | "
            f"Underpredict Rate: {metrics.get('underpredict_rate', 0):.3f}"
            f"Time: {epoch_time:.2f}s")


        if mode == "distribution":
            print(f" | Calib Error: {metrics.get('calibration_error', 0):.4f}")
        elif mode == "root_probability":
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {metrics['loss']:.4f} | "
                  f"Accuracy: {metrics.get('accuracy', 0):.4f} | "
                  f"Underpredict Rate: {metrics.get('underpredict_rate', 0):.3f} | "
                  f"Avg Under Error: {metrics.get('avg_underpredict_error', 0):.4f}")
        else:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {metrics['loss']:.4f} | "
                  f"MAE: {metrics.get('mae', 0):.4f}")
            
              
                
        # WandB logging
        # wandb.log({
        #     "epoch": epoch,
        #     "train_loss": train_loss,
        #     "val_loss": metrics["loss"],
        #     "val_mae": metrics["mae"],
        #     "val_rmse": metrics["rmse"],
        #     "val_accuracy": metrics.get("accuracy", 0)  # only for classification modes
        # })

        # Early stopping check
        early_stopping(metrics["loss"], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    except Exception as e:
        print(f"Error in epoch {epoch}: {e}")
        raise

# Final evaluation on test set
print("\nFinal Test Results:")
#test_metrics = evaluate(test_loader)
test_metrics = evaluate(model, test_loader, criterion)
print(f"Loss: {test_metrics['loss']:.4f}")

###########################################################################
        # Outlier analysis after final test evaluation
if params['enable_outlier_analysis']:
    print("\n" + "="*50)
    outlier_info = run_outlier_analysis_for_gnn(
        model=model,
        test_loader=test_loader,
        device=device,
        mode=mode
        )
            
            # Print worst cases
    print(f"\nWorst 5 outliers:")
    for i in range(min(5, len(outlier_info['indices']))):
        idx = outlier_info['indices'][i]
        print(f"Sample {idx}: True={outlier_info['true_values'][i]:.3f}, "
                    f"Pred={outlier_info['pred_values'][i]:.3f}, "
                    f"Error={outlier_info['residuals'][i]:.3f}")


    # Feature names based on your table
    feature_names = [
        'node_type',           # 0 - Node type: 0=root, 1=intermediate, 2=leaf
        'in_degree',           # 1 - Incoming edges count
        'out_degree',          # 2 - Outgoing edges count
        'betweenness',         # 3 - Betweenness centrality
        'closeness',           # 4 - Closeness centrality
        'pagerank',            # 5 - PageRank score
        'degree_centrality',   # 6 - Degree centrality
        'variable_card',       # 7 - Cardinality of the node variable
        'num_parents',         # 8 - Number of parent nodes in BN
        'evidence_flag',       # 9 - Evidence flag (0 or 1)
        'cpd_0',              # 10 - CPD value 0
        'cpd_1',              # 11 - CPD value 1
        'cpd_2',              # 12 - CPD value 2
        'cpd_3',              # 13 - CPD value 3
        'cpd_4',              # 14 - CPD value 4
        'cpd_5',              # 15 - CPD value 5
        'cpd_6',              # 16 - CPD value 6
        'cpd_7'               # 17 - CPD value 7
    ]

    structural_analysis = analyze_graph_structure_outliers(
        model=model,
        test_loader=test_loader,
        device=device,
        mode=mode
    )
        
############################################################################


if mode == "distribution":
    print(f"Accuracy: {test_metrics.get('accuracy', 0):.4f}")
elif mode == "root_probability":
    print(f"Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print(f"Underpredict Rate: {test_metrics.get('underpredict_rate', 0):.3f}")
    print(f"Average Underprediction Error: {test_metrics.get('avg_underpredict_error', 0):.4f}")
    print(f"MAE: {test_metrics.get('mae', 0):.4f}")
    print(f"RMSE: {test_metrics.get('rmse', 0):.4f}")
    print(f"Avg Time/Epoch: {np.mean(epoch_times):.2f}s")
    print(f"Total Time: {np.sum(epoch_times):.2f}s")
else:  # regression
    print(f"MAE: {test_metrics.get('mae', 0):.4f}")
    print(f"RMSE: {test_metrics.get('rmse', 0):.4f}")

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(train_losses, label="Train Loss")
plt.plot([m["loss"] for m in val_metrics], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Curve")

plt.subplot(122)
# Plot mode-specific additional metrics
if mode == "root_probability":
    plt.plot([m.get("underpredict_rate", 0) for m in val_metrics], label="Underpredict Rate")
    plt.plot([m.get("accuracy", 0) for m in val_metrics], label="Accuracy")
    plt.axhline(y=0.2, color='r', linestyle='--', label='Target Under Rate (20%)')
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Root Probability Metrics")

elif mode == "distribution":
    plt.plot([m.get("calibration_error", 0) for m in val_metrics], label="Calibration Error")
    plt.xlabel("Epoch")
    plt.ylabel("Calibration Error")
    plt.legend()
    plt.title("Calibration Error")

plt.tight_layout()
#plt.show()
plt.savefig('training_curve.png')  # Save figure to file for ssh remote access

# Plot mode-specific results
plot_results(test_loader)

# Save model with mode and strategy in filename
os.makedirs("models", exist_ok=True)
evidence_type = "intermediate" if config.get("use_intermediate") else "leaf"
model_path = f"models/graphsage_{mode}_{mask_strategy}_{evidence_type}.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Debug: Check class imbalance
print("\n=== CLASS IMBALANCE DEBUG ===")
ys = torch.cat([data.y for data in train_set])
print("Class 0:", (ys < 0.5).sum().item())
print("Class 1:", (ys >= 0.5).sum().item())
ys = torch.cat([data.y for data in train_set])
print(ys.shape)  

if __name__ == "__main__":
    # wandb.finish()
    print("Training complete. Model saved.")