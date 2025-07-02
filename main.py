import time
import os
import itertools
import pandas as pd
import numpy as np 
from config_loader import load_config
from graph_generator import generate_tree
from bayesian_network_builder import build_bn_from_tree
from exporter import (
    save_graph_with_details,
    save_cpds_as_numpy_tables,
    get_max_cpd_length,
    compute_global_max_cpd_length, 
    pad_cpd_values, 
    flatten
)
from graph_visualization import draw_graph

if __name__ == "__main__":
    start_time = time.time()
    config = load_config("config.yaml")

    # Generate all graphs and models
    all_models = []
    graph_model_pairs = []

    for i in range(config['num_graphs']):
        tree = generate_tree(config)
        model = build_bn_from_tree(tree, config)
        all_models.append(model)
        graph_model_pairs.append((tree, model))

    # Compute global CPD size
    global_max_len = compute_global_max_cpd_length(all_models)
    print(f" Global max CPD length: {global_max_len}")
    # Save global_max_len to a text file
    os.makedirs("global_datasets", exist_ok=True)
    with open("global_datasets/global_cpd_len.txt", "w") as f:
        f.write(str(global_max_len))

    # Export everything with uniform padding
    for i, (tree, model) in enumerate(graph_model_pairs):
        save_cpds_as_numpy_tables(model, os.path.join("generated_graphs", f"cpds_numpy_table_{i}.npy"))
        save_graph_with_details(tree, model, config, i, global_max_len)
        print(f" Saved Bayesian Network #{i+1}")

    end_time = time.time()
    print(f" Total execution time: {end_time - start_time:.2f} seconds")

    #cpd_tables = np.load("./generated_graphs/cpds_numpy_table_0.npy", allow_pickle=True).item()
    # print("\n CPD tables from first graph:")
    # for node, table in cpd_tables.items():
    #     print(f"\n Node {node} CPD:")
    #     print(table)
