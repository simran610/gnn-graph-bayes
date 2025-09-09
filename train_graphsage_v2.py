import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
import time
from early_stopping_pytorch import EarlyStopping
from graphsage_model import GraphSAGE
import wandb
from outlier_analysis import analyze_outliers, run_outlier_analysis_for_gnn, pick_specific_prediction_outliers
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


# Initialize data loaders
train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=params['batch_size'])
test_loader = DataLoader(test_set, batch_size=params['batch_size'])


# Load global CPD length
with open(config["global_cpd_len_path"], "r") as f:
    global_cpd_len = int(f.read())


in_channels = train_set[0].x.shape[1]


def simple_asymmetric_loss(preds, targets, penalty=1.8):
    """Simple asymmetric loss that penalizes underprediction more"""
    diff = targets - preds  # positive = underprediction, negative = overprediction
    loss = torch.where(diff > 0, penalty * diff**2, diff**2)
    return loss.mean()


# Determine model output size and loss function
if mode == "distribution":
    out_channels = 2
    loss_fn = lambda pred, true: F.kl_div(F.log_softmax(pred, dim=1), true, reduction='batchmean')
elif mode == "root_probability":
    out_channels = 1  # Single output
    loss_fn = lambda pred, true: simple_asymmetric_loss(pred, true, penalty=1.8)
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


def extract_targets_and_predictions(data, out, batch_idx=0):
    batch_size = data.batch.max().item() + 1

    if mode == "distribution":
        targets = data.y.view(batch_size, 2)
        targets = targets / targets.sum(dim=1, keepdim=True)
        predictions = out

    elif mode == "root_probability":
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

        if batch_idx == 0:
            print("Input features sample:", data.x[:5])

        optimizer.zero_grad()
        out = model(data)

        if batch_idx == 0:
            print(f"Model output sample: {out[:5]}")
            print(f"Model output shape: {out.shape}")
            print(f"Target y: {data.y[:10]}")

        targets, predictions = extract_targets_and_predictions(data, out, batch_idx)
        loss = loss_fn(predictions, targets)
        loss.backward()

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

            if predictions.dim() == 0:
                all_preds.append(predictions.cpu().unsqueeze(0))
            else:
                all_preds.append(predictions.cpu())

            if targets.dim() == 0:
                all_true.append(targets.cpu().unsqueeze(0))
            else:
                all_true.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    trues = torch.cat(all_true, dim=0).numpy()
    metrics = {
        "loss": total_loss / len(loader),
    }

    if mode == "distribution":
        softmax_preds = F.softmax(torch.tensor(preds), dim=1).numpy()
        pred_labels = np.argmax(softmax_preds, axis=1)
        true_labels = np.argmax(trues, axis=1)
        metrics["accuracy"] = np.mean(pred_labels == true_labels)

    elif mode == "root_probability":
        try:
            binary_preds = (preds > 0.5).astype(int)
            binary_trues = (trues > 0.5).astype(int)
            metrics["accuracy"] = np.mean(binary_preds == binary_trues)

            diff = trues - preds
            underpredict_mask = diff > 0
            metrics["underpredict_rate"] = np.mean(underpredict_mask)

            if np.any(underpredict_mask):
                metrics["avg_underpredict_error"] = np.mean(diff[underpredict_mask])
            else:
                metrics["avg_underpredict_error"] = 0.0

            metrics["mae"] = np.mean(np.abs(preds - trues))
            metrics["rmse"] = np.sqrt(np.mean((preds - trues) ** 2))

        except Exception as e:
            print(f"Error calculating metrics: {e}")

    else:  # regression
        metrics["mae"] = np.mean(np.abs(preds - trues))
        metrics["rmse"] = np.sqrt(np.mean((preds - trues) ** 2))

    return metrics


# Training loop with metrics tracking
train_losses = []
val_metrics = []
epoch_times = []
print("Starting training...")

for epoch in range(1, params['epochs'] + 1):
    try:
        train_loss = train()
        train_losses.append(train_loss)
        start_time = time.time()

        metrics = evaluate(model, val_loader, loss_fn)
        val_metrics.append(metrics)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {metrics['loss']:.4f} | "
            f"Underpredict Rate: {metrics.get('underpredict_rate', 0):.3f} | Time: {epoch_time:.2f}s"
        )

        if mode == "distribution":
            print(f"Calibration Error: {metrics.get('calibration_error', 0):.4f}")
        elif mode == "root_probability":
            print(
                f"Accuracy: {metrics.get('accuracy', 0):.4f} | "
                f"Avg Under Error: {metrics.get('avg_underpredict_error', 0):.4f}"
            )
        else:
            print(f"MAE: {metrics.get('mae', 0):.4f}")

        early_stopping(metrics["loss"], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    except Exception as e:
        print(f"Error in epoch {epoch}: {e}")
        raise


# Final evaluation on test set
print("\nFinal Test Results:")
test_metrics = evaluate(model, test_loader, loss_fn)
print(f"Loss: {test_metrics['loss']:.4f}")


# ======== TEMPERATURE SCALING ======== #
if mode in ["root_probability", "distribution"]:
    print("\n=== Temperature Scaling Calibration & Evaluation ===")
    temp_scaler = TemperatureScaling(init_temp=1.5)
    optimal_T = temp_scaler.calibrate(model, val_loader, device, mode=mode)
    print(f"Optimal temperature: {optimal_T:.4f}")
    calibrated_preds, targets = temp_scaler.apply(model, test_loader, device, mode=mode)
    if calibrated_preds is not None:
        mae = np.mean(np.abs(calibrated_preds.numpy() - targets.numpy()))
        print(f"Temperature-scaled Test MAE: {mae:.4f}")
        np.save("calibrated_test_probs.npy", calibrated_preds.numpy())
        np.save("calibrated_test_targets.npy", targets.numpy())


# Outlier analysis after final test evaluation
if params['enable_outlier_analysis']:
    print("\n" + "=" * 50)
    outlier_info = run_outlier_analysis_for_gnn(
        model=model,
        test_loader=test_loader,
        device=device,
        mode=mode
    )

    print(f"\nWorst 5 outliers:")
    for i in range(min(5, len(outlier_info['indices']))):
        idx = outlier_info['indices'][i]
        print(
            f"Sample {idx}: True={outlier_info['true_values'][i]:.3f}, "
            f"Pred={outlier_info['pred_values'][i]:.3f}, "
            f"Error={outlier_info['residuals'][i]:.3f}"
        )

    feature_names = [
        'node_type',          # 0
        'in_degree',          # 1
        'out_degree',         # 2
        'betweenness',        # 3
        'closeness',          # 4
        'pagerank',           # 5
        'degree_centrality',  # 6
        'variable_card',      # 7
        'num_parents',        # 8
        'evidence_flag',      # 9
        'cpd_0',              # 10
        'cpd_1',              # 11
        'cpd_2',              # 12
        'cpd_3',              # 13
        'cpd_4',              # 14
        'cpd_5',              # 15
        'cpd_6',              # 16
        'cpd_7'               # 17
    ]

    specific_outliers = pick_specific_prediction_outliers(
        y_true=outlier_info['true_values'],
        y_pred=outlier_info['pred_values'],
        X_features=outlier_info['features'],
        feature_names=feature_names,
        n_each=2
    )


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
plt.savefig('training_curve.png')


# Plot mode-specific results
plot_results(test_loader)


# Save model
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
    print("Training complete. Model saved.")
