import torch

def preprocess_graph(data, global_cpd_len, mode="regression"):
    x = data.x.clone()
    cpd_start_idx = x.size(1) - global_cpd_len  # CPD starts here
    
    # Find root node(s) by node type == 0
    node_types = x[:, 0]
    root_indices = (node_types == 0).nonzero(as_tuple=False).squeeze()
    
    if root_indices.numel() == 0:
        raise ValueError("No root node found in the graph.")
    
    # Assuming single root node
    root_node = root_indices.item() if root_indices.dim() == 0 else root_indices[0].item()

    if mode == "regression":
        y = x[root_node, cpd_start_idx:].clone()
        x[root_node, cpd_start_idx:] = 0.0
        data.y = y

    # Classification mode later ------ REMOVE IF NOT NEEDED ------
    # elif mode == "root_classification":
    #     full_root_cpd = x[root_node, cpd_start_idx:]
    #     y = torch.argmax(full_root_cpd)
    #     x[root_node, cpd_start_idx:] = 0.0
    #     data.y = y

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    data.x = x
    return data
