# File: main.py
import time
import numpy as np 
from config_loader import load_config
from graph_generator import generate_tree
from bayesian_network_builder import build_bn_from_tree
from exporter import save_graph, save_graph_with_details, save_cpds_as_numpy_tables 
from graph_visualization import draw_graph

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    config = load_config("config.yaml")
    for i in range(config['num_graphs']):
        tree = generate_tree(config)
        bn = build_bn_from_tree(tree, config)

        save_cpds_as_numpy_tables(bn, f"./generated_graphs/cpds_numpy_table_{i}.npy")

        # save_graph(tree, config, i)
        save_graph_with_details(tree, bn, config, i)
        print(f"Generated Bayesian Network #{i+1}")

        draw_graph(tree, f"./generated_graphs/graph_{i}.png")
        
    end_time = time.time()  # Record the end time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

# Load and view the CPD tables saved as a .npy file
cpd_tables = np.load(f"./generated_graphs/cpds_numpy_table_0.npy", allow_pickle=True).item()

# Print all node CPDs
print("\nAll CPD tables:")
for node, table in cpd_tables.items():
   print(f"\nNode {node} CPD:")
   print(table)