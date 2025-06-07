import torch
import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.preprocess import preprocess_graph

# Load saved dataset
dataset_path = "saved_datasets/bn_pyg_dataset.pt"
global_cpd_path = "saved_datasets/global_cpd_len.txt"

dataset = torch.load(dataset_path, weights_only=False)
print(f"Loaded dataset with {len(dataset)} graphs")

with open(global_cpd_path, "r") as f:
    global_cpd_len = int(f.read().strip())
print(f"Loaded global CPD length: {global_cpd_len}")

# Preprocess all graphs in regression mode
processed_dataset = [preprocess_graph(g, global_cpd_len, mode="regression") for g in dataset]
# random.shuffle(processed_dataset)

# Split dataset into train/val/test
total = len(processed_dataset)
train_len = int(0.8 * total)
val_len = int(0.1 * total)
test_len = total - train_len - val_len

train_set = processed_dataset[:train_len]
val_set = processed_dataset[train_len:train_len + val_len]
test_set = processed_dataset[train_len + val_len:]

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# Save splits
os.makedirs("saved_datasets", exist_ok=True)
torch.save(train_set, "saved_datasets/train.pt")
torch.save(val_set, "saved_datasets/val.pt")
torch.save(test_set, "saved_datasets/test.pt")
print("Dataset splits saved successfully.")
