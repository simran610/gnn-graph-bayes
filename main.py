# import time
# import os
# import itertools
# import pandas as pd
# import numpy as np 
# from config_loader import load_config
# from graph_generator import generate_tree
# from bayesian_network_builder import build_bn_from_tree
# from exporter import (
#     save_graph_with_details,
#     save_cpds_as_numpy_tables,
#     get_max_cpd_length,
#     compute_global_max_cpd_length, 
#     pad_cpd_values, 
#     flatten
# )
# from graph_visualization import draw_graph
# from bayesian_network_generator import generate_varied_bn

# if __name__ == "__main__":
#     start_time = time.time()
#     config = load_config("config.yaml")

#     # Generate all graphs and models
#     all_models = []
#     graph_model_pairs = []

#     # old code with bn build nad generated_graphs
#     # for i in range(config['num_graphs']):
#     #     tree = generate_tree(config)
#     #     model = build_bn_from_tree(tree, config)
#     #     all_models.append(model)
#     #     graph_model_pairs.append((tree, model))

#     # New code with varied BN generator
#     for i in range(config['num_graphs']):
#         tree, model = generate_varied_bn(config)  
#         all_models.append(model)
#         graph_model_pairs.append((tree, model))

#     # Compute global CPD size
#     global_max_len = compute_global_max_cpd_length(all_models)
#     print(f" Global max CPD length: {global_max_len}")
#     # Save global_max_len to a text file
#     os.makedirs("global_datasets", exist_ok=True)
#     with open("global_datasets/global_cpd_len.txt", "w") as f:
#         f.write(str(global_max_len))

#     # Export everything with uniform padding
#     for i, (tree, model) in enumerate(graph_model_pairs):
#         save_cpds_as_numpy_tables(model, os.path.join("generated_graphs", f"cpds_numpy_table_{i}.npy"))
#         save_graph_with_details(tree, model, config, i, global_max_len)
#         print(f" Saved Bayesian Network #{i+1}")

#     end_time = time.time()
#     print(f" Total execution time: {end_time - start_time:.2f} seconds")

#     #cpd_tables = np.load("./generated_graphs/cpds_numpy_table_0.npy", allow_pickle=True).item()
#     # print("\n CPD tables from first graph:")
#     # for node, table in cpd_tables.items():
#     #     print(f"\n Node {node} CPD:")
#     #     print(table)


import time
import os
import itertools
import pandas as pd
import numpy as np 
from multiprocessing import Pool, cpu_count
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
from bayesian_network_generator import generate_varied_bn_fixed_cycle_check


def generate_single_graph(args):
    """Worker function for multiprocessing"""
    i, config = args
    tree, model = generate_varied_bn_fixed_cycle_check(config)
    if i % 100 == 0:
        print(f"Generated graph {i}")
    return (tree, model)


if __name__ == "__main__":
    start_time = time.time()
    config = load_config("config.yaml")

    # Prepare configs with different random seeds
    configs_with_seeds = [
        (i, {**config, 'random_seed': i}) 
        for i in range(config['num_graphs'])
    ]

    # Generate all graphs and models using multiprocessing
    print(f"Generating {config['num_graphs']} graphs using {cpu_count()-1} processes...")
    
    with Pool(processes=cpu_count() - 1) as pool:
        graph_model_pairs = pool.map(generate_single_graph, configs_with_seeds)
    
    all_models = [model for _, model in graph_model_pairs]
    
    print(f"Graph generation complete! Time: {time.time() - start_time:.2f}s")

    # Compute global CPD size
    global_max_len = compute_global_max_cpd_length(all_models)
    print(f"Global max CPD length: {global_max_len}")
    
    # Save global_max_len to a text file
    os.makedirs("global_datasets", exist_ok=True)
    with open("global_datasets/global_cpd_len.txt", "w") as f:
        f.write(str(global_max_len))

    # Export everything with uniform padding
    print("Saving graphs...")
    for i, (tree, model) in enumerate(graph_model_pairs):
        save_cpds_as_numpy_tables(model, os.path.join("generated_graphs", f"cpds_numpy_table_{i}.npy"))
        save_graph_with_details(tree, model, config, i, global_max_len)
        if (i + 1) % 100 == 0:
            print(f"Saved {i+1} graphs")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")