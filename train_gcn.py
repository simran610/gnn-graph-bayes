# Train_gcn.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
from early_stopping_pytorch import EarlyStopping
from gcn_model import GCN

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
    'epochs': int(config.get("epochs", 100))
}

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

# Debug: Check class imbalance
print("\n=== CLASS IMBALANCE DEBUG ===")
ys = torch.cat([data.y for data in train_set])
print("Class 0:", (ys < 0.5).sum().item())
print("Class 1:", (ys >= 0.5).sum().item())

# Load global CPD length
with open(config["global_cpd_len_path"], "r") as f:
    global_cpd_len = int(f.read())

in_channels = train_set[0].x.shape[1]

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
    loss_fn = torch.nn.MSELoss()
else:  # regression
    out_channels = global_cpd_len
    loss_fn = torch.nn.MSELoss()

print(f"Model config - Input: {in_channels}, Output: {out_channels}")

# Initialize model with proper device placement
model = GCN(
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
        targets = data.y.view(-1)
        predictions = out.view(-1)
       
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
            debug_batch(data, batch_idx)
        
        optimizer.zero_grad()
        out = model(data)
        
        # Debug model output
        if batch_idx == 0:
            print(f"Model output shape: {out.shape}")
            print(f"Model output: {out}")
            print(f"Model output (sigmoid probabilities): {torch.sigmoid(out)}")
        
        # Extract targets and predictions properly
        targets, predictions = extract_targets_and_predictions(data, out, batch_idx)
        
        # Compute loss
        loss = loss_fn(predictions, targets)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

#old: evaluate function
# def evaluate(loader):
#     """Evaluation with mode-specific metrics"""
#     model.eval()
#     total_loss = 0
#     all_preds = []
#     all_true = []
    
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             out = model(data)
            
#             # Extract targets and predictions properly
#             targets, predictions = extract_targets_and_predictions(data, out)
            
#             # Compute loss
#             loss = loss_fn(predictions, targets)
#             total_loss += loss.item()
            
#             # Store predictions and targets for metrics calculation
#             if mode == "distribution":
#                 all_preds.append(F.softmax(predictions, dim=1))
#                 all_true.append(targets)
#             elif mode == "root_probability":
#                 all_preds.append(torch.sigmoid(predictions))
#                 all_true.append(targets)
           
    
#     # Calculate mode-specific metrics
#     metrics = {"loss": total_loss / len(loader)}
    
#     if mode == "distribution" and all_preds:
#         preds = torch.cat(all_preds)
#         trues = torch.cat(all_true)
#         # Calculate calibration error using the positive class probability
#         pred_labels = preds.argmax(dim=1)
#         true_labels = trues.argmax(dim=1)
#         metrics["accuracy"] = (pred_labels == true_labels).float().mean().item()
#         # metrics["calibration_error"] = (preds[:,1] - trues[:,1]).abs().mean().item()
        
#     elif mode == "root_probability" and all_preds:
#         preds = torch.cat(all_preds)
#         trues = torch.cat(all_true)
        
#         # Calculate accuracy
#         # binary_preds = (preds > 0.5).float()
#         # metrics["accuracy"] = (binary_preds == trues).float().mean().item()
#         binary_preds = (preds > 0.5).float()
#         binary_trues = (trues > 0.5).float()
#         metrics["accuracy"] = (binary_preds == binary_trues).float().mean().item()
        
#         # Calculate AUC
#     try:
#         # Convert continuous targets to binary if needed
#         trues_np = trues.cpu().numpy()
#         if not np.all(np.isin(trues_np, [0, 1])):
#             trues_np = (trues_np > 0.5).astype(int)
#         metrics["auc"] = roc_auc_score(trues_np, preds.cpu().numpy())
#     except Exception as e:
#         print(f"Warning: Could not calculate AUC: {e}")
#         metrics["auc"] = 0.5  # Fallback if AUC calculation fails
    
#     return metrics
# NEw: Enhanced evaluation metrics

def evaluate(loader):
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
            
            all_preds.append(predictions.cpu())
            all_true.append(targets.cpu())
    
    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_true).numpy()
    metrics = {
        "loss": total_loss / len(loader),
        "mae": np.mean(np.abs(preds - trues)),
        "rmse": np.sqrt(np.mean((preds - trues)**2))
    }
    
    if mode == "distribution":
        softmax_preds = F.softmax(torch.tensor(preds), dim=1).numpy()
        pred_labels = np.argmax(softmax_preds, axis=1)
        true_labels = np.argmax(trues, axis=1)
        metrics["accuracy"] = np.mean(pred_labels == true_labels)
        
    elif mode == "root_probability":
        binary_preds = (preds > 0.5).astype(int)
        binary_trues = (trues > 0.5).astype(int)
        metrics["binary_accuracy"] = np.mean(binary_preds == binary_trues)
    
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

#Old: plot_root_probability_results
# def plot_root_probability_results(loader):
#     """Plot confusion matrix for root_probability mode"""
#     preds, trues = [], []
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             out = model(data)
#             targets, predictions = extract_targets_and_predictions(data, out)
            
#             binary_preds = (torch.sigmoid(predictions) > 0.5).float()
#             preds.append(binary_preds)
#             trues.append(targets)
    
#     preds = torch.cat(preds).cpu().numpy().astype(int)  # Convert to integers
#     trues = torch.cat(trues).cpu().numpy()
    
#     # Convert continuous targets to binary if needed
#     if not np.all(np.isin(trues, [0, 1])):
#         binary_trues = (trues > 0.5).astype(int)
#     else:
#         binary_trues = trues.astype(int)
    
#     cm = confusion_matrix(binary_trues, preds)
#     plt.figure(figsize=(6,6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=['Class 0', 'Class 1'], 
#                 yticklabels=['Class 0', 'Class 1'])
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix")
#     plt.show()

# NEW: plot_root_probability_results
def plot_root_probability_results(loader):
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            targets, predictions = extract_targets_and_predictions(data, out)
            preds.append(predictions.cpu())
            trues.append(targets.cpu())
    
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
    plt.show()


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

print("Starting training...")

for epoch in range(1, params['epochs'] + 1):
    # Training phase
    try:
        train_loss = train()
        train_losses.append(train_loss)
        
        # Validation phase
        metrics = evaluate(val_loader)
        val_metrics.append(metrics)
        
        # Print epoch statistics
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {metrics['loss']:.4f}", end="")
        
        if mode == "distribution":
            print(f" | Calib Error: {metrics.get('calibration_error', 0):.4f}")
        elif mode == "root_probability":
            print(f" | Acc: {metrics.get('accuracy', 0):.4f} | AUC: {metrics.get('auc', 0):.4f}")
        else:
            print()
        
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
test_metrics = evaluate(test_loader)
print(f"Loss: {test_metrics['loss']:.4f}")
print(f"MAE: {test_metrics['mae']:.4f}")
print(f"RMSE: {test_metrics['rmse']:.4f}")
if mode == "distribution":
    # print(f"Calibration Error: {test_metrics.get('calibration_error', 0):.4f}")
    print(f"Accuracy: {test_metrics.get('accuracy', 0):.4f}")
elif mode == "root_probability":
    print(f"Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print(f"AUC: {test_metrics.get('auc', 0):.4f}")

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
    plt.plot([m.get("accuracy", 0) for m in val_metrics], label="Accuracy")
    plt.plot([m.get("auc", 0.5) for m in val_metrics], label="AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Root_probability Metrics")
elif mode == "distribution":
    plt.plot([m.get("calibration_error", 0) for m in val_metrics], label="Calibration Error")
    plt.xlabel("Epoch")
    plt.ylabel("Calibration Error")
    plt.legend()
    plt.title("Calibration Error")

plt.tight_layout()
plt.show()

# Plot mode-specific results
plot_results(test_loader)

# Save model with mode and strategy in filename
os.makedirs("models", exist_ok=True)
model_path = f"models/gcn_{mode}_{mask_strategy}.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Debug: Check class imbalance
print("\n=== CLASS IMBALANCE DEBUG ===")
ys = torch.cat([data.y for data in train_set])
print("Class 0:", (ys < 0.5).sum().item())
print("Class 1:", (ys >= 0.5).sum().item())
ys = torch.cat([data.y for data in train_set])
print(ys.shape)  
