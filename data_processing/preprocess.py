import torch
import random
import yaml
from pgmpy.inference import VariableElimination
from build_pgmpy_bn_from_json import build_pgmpy_bn_from_json

# Load config and random seed
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
random.seed(config.get("random_seed", 42))
json_folder = config.get("output_dir", "./generated_graphs" )

def preprocess_graph(data, global_cpd_len, mode="regression", num_leaf_to_infer=None):
    x = data.x.clone()

    # Feature indices
    node_type_idx = 0
    var_card_idx = 7
    cpd_start_idx = 9

    # Identify node types
    node_types = x[:, node_type_idx]
    root_indices = (node_types == 0).nonzero(as_tuple=False).squeeze()
    leaf_indices = (node_types == 2).nonzero(as_tuple=False).squeeze()

    if root_indices.numel() == 0:
        raise ValueError("No root node found.")
    if leaf_indices.numel() == 0:
        raise ValueError("No leaf nodes found.")

    root_node = root_indices.item() if root_indices.dim() == 0 else root_indices[0].item()

    if mode == "regression":
        # Predict full root CPD
        y = x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len].clone()
        x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len] = 0.0

        data.y = y
        data.root_node = root_node
        data.leaf_nodes = leaf_indices

    elif mode == "conditional_probability":
        if num_leaf_to_infer is None:
            num_leaf_to_infer = config.get("num_leaf_to_infer", 2)

        # Build PGMPY model from PyG graph
        bn_model, var_name_map = build_pgmpy_bn_from_json(json_folder)
        infer = VariableElimination(bn_model)

        # Select random subset of leaf nodes to use as evidence
        num_to_condition = min(num_leaf_to_infer, leaf_indices.numel())
        conditioned_leaf_nodes = random.sample(leaf_indices.tolist(), num_to_condition)

        evidence_dict = {}
        for leaf in conditioned_leaf_nodes:
            card = int(x[leaf, var_card_idx].item())
            state = random.randint(0, card - 1)
            evidence_dict[var_name_map[leaf]] = state

        # Perform inference to get conditional P(root | evidence)
        q = infer.query(
            variables=[var_name_map[root_node]],
            evidence=evidence_dict,
            show_progress=False
        )
        y = torch.tensor(q.values, dtype=torch.float)

        # Mask CPD of root node in feature matrix
        x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len] = 0.0
        for leaf in conditioned_leaf_nodes:
            x[leaf, cpd_start_idx:cpd_start_idx + global_cpd_len] = 0.0

        data.y = y
        data.root_node = root_node
        data.conditioned_leaf_nodes = torch.tensor(conditioned_leaf_nodes, dtype=torch.long)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    data.x = x
    return data
