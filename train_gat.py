import torch
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from gat_model import GAT
from early_stopping_pytorch import EarlyStopping
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#'Values tried '------------------------
#'epoch 1-101 '
#'early stopping patience 5 '
#'learning rate 0.001 '  ----- 0.0001
#'weight decay 5e-4 '---- 5e-4
#'droupout 0.5 '---- 0.8
#'batch size 32 '
#'hidden channels 32 '

# Initialize early stopping object
early_stopping = EarlyStopping(patience=5, verbose=True)

# Load datasets
train_set = torch.load("saved_datasets/train.pt", weights_only=False)
val_set = torch.load("saved_datasets/val.pt", weights_only=False)
test_set = torch.load("saved_datasets/test.pt", weights_only=False)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

# Load global CPD length (regression output size)
with open("saved_datasets/global_cpd_len.txt", "r") as f:
    global_cpd_len = int(f.read().strip())

# Use first graph to get input feature size
sample_data = train_set[0]
in_channels = sample_data.x.size(1)
out_channels = global_cpd_len
loss_fn = MSELoss()

# Initialize model and optimizer
model = GAT(in_channels, 32, out_channels, dropout=0.8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

# Evaluates model
def evaluate(loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            y = torch.stack([g.y for g in data.to_data_list()]).to(device).float()
            loss = loss_fn(out, y)
            total_loss += loss.item()

    return total_loss / len(loader)

# Plots true vs predicted values
def scatter_true_vs_pred(loader, title="True vs Predicted (Validation Set)"):
    model.eval()
    true_vals = []
    pred_vals = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            y = torch.stack([g.y for g in data.to_data_list()]).to(device).float()
            true_vals.append(y.cpu())
            pred_vals.append(out.cpu())

    true_vals = torch.cat(true_vals, dim=0).numpy().flatten()
    pred_vals = torch.cat(pred_vals, dim=0).numpy().flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.5, label="Predicted vs True")
    plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--', label='Ideal Line')
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Training Loop
train_losses = []
val_losses = []

for epoch in range(1, 101):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = torch.stack([g.y for g in data.to_data_list()]).to(device).float()
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # saving losses
    train_loss_avg = total_loss / len(train_loader)
    val_loss = evaluate(val_loader)

    train_losses.append(train_loss_avg)
    val_losses.append(val_loss)
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

    print(f"Epoch {epoch:03d}: Train Loss = {train_loss_avg:.4f}, Val Loss = {val_loss:.4f}")

# Test Evaluation
test_loss = evaluate(test_loader)
print(f"\nFinal Test Loss: {test_loss:.4f}")

# Plot 1: Training vs Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss", color="blue", linewidth=2)
plt.plot(val_losses, label="Validation Loss", color="orange", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: True vs Predicted Plot
scatter_true_vs_pred(val_loader)

# Save Model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/gat_model_regression.pt")
