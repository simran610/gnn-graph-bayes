import os
import json
import numpy as np

# Set paths
json_dir = "./generated_graphs"
npy_dir = "./generated_graphs"

# How many files to check
max_files = 5
tolerance = 1e-6
mismatch_count = 0

for i in range(max_files):
    json_path = os.path.join(json_dir, f"detailed_graph_{i}.json")
    npy_path = os.path.join(npy_dir, f"cpds_numpy_table_{i}.npy")

    if not os.path.exists(json_path) or not os.path.exists(npy_path):
        print(f"[Skipping] Missing: {json_path} or {npy_path}")
        continue

    # Load both CPD sources
    with open(json_path, "r") as f:
        json_data = json.load(f)
    npy_cpds = np.load(npy_path, allow_pickle=True).item()

    for node_id, node_info in json_data["nodes"].items():
        node_id_str = str(node_id)
        json_cpd = np.array(node_info["cpd"]["values"])
        npy_df = npy_cpds.get(node_id_str)

        if npy_df is None:
            print(f"[Missing] Node {node_id_str} not in .npy for graph {i}")
            mismatch_count += 1
            continue

        # Extract probability column from DataFrame
        if hasattr(npy_df, "columns") and "P" in npy_df.columns:
            npy_cpd = npy_df["P"].values
        else:
            print(f"[No 'P' column] Node {node_id_str} in .npy for graph {i}")
            mismatch_count += 1
            continue

        # Ensure 2D shape for normalization check
        json_cpd_2d = np.atleast_2d(json_cpd)
        npy_cpd_2d = np.atleast_2d(npy_cpd)

        # --- Normalization check for JSON CPD ---
        json_row_sums = np.sum(json_cpd_2d, axis=1)
        if not np.allclose(json_row_sums, 1.0, atol=tolerance):
            print(f"[Not Normalized] JSON CPD for Graph {i}, Node {node_id}: row sums = {json_row_sums}")
            mismatch_count += 1

        # --- Normalization check for NPY CPD ---
        npy_row_sums = np.sum(npy_cpd_2d, axis=1)
        if not np.allclose(npy_row_sums, 1.0, atol=tolerance):
            print(f"[Not Normalized] NPY CPD for Graph {i}, Node {node_id}: row sums = {npy_row_sums}")
            mismatch_count += 1

        # Flatten for comparison (column-major order)
        json_flat = json_cpd_2d.flatten(order="F")
        npy_flat = npy_cpd_2d.flatten(order="F")

        if json_flat.shape != npy_flat.shape:
            print(f"[Shape Mismatch] Graph {i}, Node {node_id}: {json_flat.shape} vs {npy_flat.shape}")
            mismatch_count += 1
            continue

        if not np.allclose(json_flat, npy_flat, atol=tolerance):
            print(f"[Mismatch] Graph {i}, Node {node_id}: values differ beyond tolerance")
            mismatch_count += 1

print(f"\n✅ Finished comparing {max_files} graphs.")
if mismatch_count == 0:
    print("✅ All CPDs match and are normalized!")
else:
    print(f"❌ Found {mismatch_count} mismatches or normalization errors.")