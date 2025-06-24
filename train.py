# import torch
# import random
# import os
# import sys
# import yaml

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from data_processing.preprocess import preprocess_graph

# random.seed(42)

# dataset_path = "saved_datasets/bn_pyg_dataset.pt"
# global_cpd_path = "saved_datasets/global_cpd_len.txt"

# dataset = torch.load(dataset_path, weights_only=False)
# print(f"Loaded dataset with {len(dataset)} graphs")

# with open(global_cpd_path, "r") as f:
#     global_cpd_len = int(f.read().strip())
# print(f"Loaded global CPD length: {global_cpd_len}")

# # Configurable params
# mode = "conditional_probability"  # or "regression"
# num_leaf_to_condition = 3  # customize as needed

# processed_dataset = [preprocess_graph(g, global_cpd_len, mode=mode, num_leaf_to_condition=num_leaf_to_condition) for g in dataset]

# random.shuffle(processed_dataset)

# total = len(processed_dataset)
# train_len = int(0.8 * total)
# val_len = int(0.1 * total)
# test_len = total - train_len - val_len

# train_set = processed_dataset[:train_len]
# val_set = processed_dataset[train_len:train_len + val_len]
# test_set = processed_dataset[train_len + val_len:]

# print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# os.makedirs("saved_datasets", exist_ok=True)
# torch.save(train_set, "saved_datasets/train.pt")
# torch.save(val_set, "saved_datasets/val.pt")
# torch.save(test_set, "saved_datasets/test.pt")
# print("Dataset splits saved successfully.")

import yaml
import random
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.preprocess import preprocess_graph

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

random.seed(config.get("random_seed", 42))

dataset_path = config["dataset_path"]
global_cpd_len_path = config["global_cpd_len_path"]

dataset = torch.load(dataset_path, weights_only=False)
print(f"Loaded dataset with {len(dataset)} graphs")

with open(global_cpd_len_path, "r") as f:
    global_cpd_len = int(f.read().strip())
print(f"Loaded global CPD length: {global_cpd_len}")

mode = config.get("mode", "regression")
num_leaf_to_infer = config.get("num_leaf_to_infer", 2)

processed_dataset = [preprocess_graph(
    g,
    global_cpd_len,
    mode=mode,
    num_leaf_to_infer=num_leaf_to_infer
) for g in dataset]

random.shuffle(processed_dataset)

total = len(processed_dataset)
train_len = int(config.get("train_split", 0.8) * total)
val_len = int(config.get("val_split", 0.1) * total)
test_len = total - train_len - val_len

train_set = processed_dataset[:train_len]
val_set = processed_dataset[train_len:train_len + val_len]
test_set = processed_dataset[train_len + val_len:]

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

os.makedirs("saved_datasets", exist_ok=True)
torch.save(train_set, "saved_datasets/train.pt")
torch.save(val_set, "saved_datasets/val.pt")
torch.save(test_set, "saved_datasets/test.pt")
print("Dataset splits saved successfully.")
