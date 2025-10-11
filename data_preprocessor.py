"""
Node Feature Indexing (consistent across BN→NX→PyG conversion and preprocessing):

Index | Feature             | Description
-----------------------------------------------------------
0     | node_type           | Node type: 0=root, 1=intermediate, 2=leaf
1     | in_degree           | Incoming edges count
2     | out_degree          | Outgoing edges count
3     | betweenness         | Betweenness centrality
4     | closeness           | Closeness centrality
5     | pagerank            | PageRank score
6     | degree_centrality   | Degree centrality
7     | variable_card       | Cardinality of the node variable
8     | num_parents         | Number of parent nodes in BN
9     | evidence_flag       | Evidence flag (added during preprocessing; 0 or 1)
10+    | CPD values         | Flattened Conditional Probability Table values

Added features appended at end of node features vector:
- CPD Entropy
- Evidence Strength (copy of evidence flag)
- Distance to Nearest Evidence

Important:
- Evidence flag is inserted at index 9 during preprocessing shifting CPD start to index 10.
- Normalization applies to indices 2-6.
- Masking applies to CPD feature columns starting at index 10.
- New features added after original features.

Note: Comments created using chatGPT-4, based on the provided context and code structure.
"""

import os
import json
import random
import torch
import yaml
import scipy.stats
import numpy as np
from tqdm import tqdm
#from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
import networkx as nx


class BayesianNetworkBuilder:
    """Builds Bayesian networks from JSON with error handling"""

    def __init__(self, normalize_cpds=False, verbose=False):
        self.normalize_cpds = normalize_cpds
        self.verbose = verbose

    def build_from_json(self, filepath):
        """Build pgmpy BN from JSON file"""
        with open(filepath, "r") as f:
            json_data = json.load(f)

        # model = DiscreteBayesianNetwork()
        model = BayesianNetwork()

        # Add edges
        for edge in json_data["edges"]:
            try:
                model.add_edge(str(edge["source"]), str(edge["target"]))
            except Exception as e:
                if self.verbose:
                    print(f"Error adding edge {edge}: {e}")

        # Add CPDs
        cpd_errors = 0
        for node_id, node_info in json_data["nodes"].items():
            try:
                cpd = self._create_cpd(node_id, node_info)
                model.add_cpds(cpd)
            except Exception as e:
                cpd_errors += 1
                if self.verbose:
                    print(f"Error creating CPD for node {node_id}: {e}")
                cpd = self._create_uniform_cpd(node_id, node_info)
                model.add_cpds(cpd)

        if cpd_errors > 0 and self.verbose:
            print(f"Graph had {cpd_errors} CPD errors - using uniform fallbacks")

        try:
            model.check_model()
        except Exception as e:
            if self.verbose:
                print(f"Model validation failed: {e}")

        return model, json_data

    def _create_uniform_cpd(self, node_id, node_info):
        """Create uniform CPD as fallback"""
        cpd_info = node_info["cpd"]
        variable_card = cpd_info["variable_card"]
        evidence = [str(e) for e in cpd_info.get("evidence", [])]
        evidence_card = cpd_info.get("evidence_card", [])

        uniform_prob = 1.0 / variable_card

        if evidence:
            num_cols = int(np.prod(evidence_card))
            values_matrix = [[uniform_prob] * num_cols for _ in range(variable_card)]
        else:
            values_matrix = [[uniform_prob] for _ in range(variable_card)]

        return TabularCPD(
            variable=str(node_id),
            variable_card=variable_card,
            values=values_matrix,
            evidence=evidence if evidence else None,
            evidence_card=evidence_card if evidence_card else None
        )

    def _create_cpd(self, node_id, node_info):
        """Create CPD from node info"""
        cpd_info = node_info["cpd"]
        variable_card = cpd_info["variable_card"]
        values = cpd_info["values"]
        evidence = [str(e) for e in cpd_info.get("evidence", [])]
        evidence_card = cpd_info.get("evidence_card", [])

        if evidence:
            expected_cols = int(np.prod(evidence_card))
            expected_length = variable_card * expected_cols

            if len(values) != expected_length:
                if self.verbose:
                    print(f"Node {node_id}: Adjusting values length from {len(values)} to {expected_length}")

                if len(values) < expected_length:
                    pad_value = 1.0 / variable_card
                    values.extend([pad_value] * (expected_length - len(values)))
                else:
                    values = values[:expected_length]

            try:
                values_matrix = np.array(values).reshape(variable_card, expected_cols).tolist()
            except ValueError:
                uniform_val = 1.0 / variable_card
                values_matrix = [[uniform_val] * expected_cols for _ in range(variable_card)]
        else:
            if len(values) < variable_card:
                values.extend([1.0 / variable_card] * (variable_card - len(values)))
            values_matrix = [[values[i]] for i in range(variable_card)]

        if self.normalize_cpds:
            values_matrix = self._normalize_matrix(values_matrix, variable_card)

        return TabularCPD(
            variable=str(node_id),
            variable_card=variable_card,
            values=values_matrix,
            evidence=evidence if evidence else None,
            evidence_card=evidence_card if evidence_card else None
        )

    def _normalize_matrix(self, values_matrix, variable_card):
        """Normalize matrix so each column sums to 1"""
        num_cols = len(values_matrix[0])
        normalized_matrix = []

        for var_state in range(variable_card):
            normalized_row = []
            for col in range(num_cols):
                col_sum = sum(values_matrix[i][col] for i in range(variable_card))
                normalized_value = values_matrix[var_state][col] / col_sum if col_sum != 0 else 1.0 / variable_card
                normalized_row.append(normalized_value)
            normalized_matrix.append(normalized_row)

        return normalized_matrix


class InferenceEngine:
    """Handles probabilistic inference with separate leaf/intermediate evidence tracking"""

    def __init__(self, num_leaf_to_infer=2, num_intermediate_to_infer=2, 
                 query_state=None, verbose=False):
        self.num_leaf_to_infer = num_leaf_to_infer
        self.num_intermediate_to_infer = num_intermediate_to_infer
        self.query_state = query_state
        self.verbose = verbose
        if self.verbose:
            print(f"InferenceEngine initialized: num_leaf_to_infer={self.num_leaf_to_infer}, "
                  f"num_intermediate_to_infer={self.num_intermediate_to_infer}, "
                  f"query_state={self.query_state}")

    def infer_root_given_mixed_evidence(self, model, json_data, use_intermediate=True):
        """
        Select evidence from BOTH intermediate AND leaf nodes.
        This combines evidence from multiple node types for richer inference.
        """
        infer = VariableElimination(model)
        root_node = str(json_data["node_types"]["roots"][0])
        
        evidence = {}
        selected_nodes = []
        
        # === SELECT INTERMEDIATE NODES ===
        if use_intermediate:
            intermediate_nodes = json_data["node_types"].get("intermediates", [])
            if intermediate_nodes:
                num_to_select = min(len(intermediate_nodes), self.num_intermediate_to_infer)
                conditioned_intermediate = random.sample(intermediate_nodes, num_to_select)
                selected_nodes.extend(conditioned_intermediate)
                
                for node in conditioned_intermediate:
                    state = self.query_state if self.query_state is not None else random.randint(0, 1)
                    evidence[str(node)] = state
                
                if self.verbose:
                    print(f"Selected {len(conditioned_intermediate)} intermediate nodes: {conditioned_intermediate}")
        
        # === SELECT LEAF NODES ===
        leaf_nodes = json_data["node_types"]["leaves"]
        if leaf_nodes:
            num_to_select = min(len(leaf_nodes), self.num_leaf_to_infer)
            conditioned_leaves = random.sample(leaf_nodes, num_to_select)
            selected_nodes.extend(conditioned_leaves)
            
            for leaf in conditioned_leaves:
                state = self.query_state if self.query_state is not None else random.randint(0, 1)
                evidence[str(leaf)] = state
            
            if self.verbose:
                print(f"Selected {len(conditioned_leaves)} leaf nodes: {conditioned_leaves}")
        
        if self.verbose:
            print(f"Total evidence nodes selected: {len(selected_nodes)} from both intermediate and leaf")
        
        try:
            q = infer.query(variables=[root_node], evidence=evidence, show_progress=False)
            return torch.tensor(q.values, dtype=torch.float), evidence
        except Exception as e:
            if self.verbose:
                print(f"Inference failed (mixed evidence): {e}")
            return torch.tensor([0.5, 0.5], dtype=torch.float), evidence

    def infer_root_given_intermediate(self, model, json_data):
        """Perform inference from intermediate to root nodes (LEAF ONLY - deprecated)"""
        infer = VariableElimination(model)
        root_node = str(json_data["node_types"]["roots"][0])
        intermediate_nodes = json_data["node_types"].get("intermediates", [])

        if not intermediate_nodes:
            if self.verbose:
                print("No intermediate nodes found, falling back to leaf-based inference")
            return self.infer_root_given_leaf(model, json_data)

        conditioned = random.sample(
            intermediate_nodes, min(len(intermediate_nodes), self.num_intermediate_to_infer)
        )

        if self.verbose:
            print(f"Selected conditioned intermediate nodes (count={len(conditioned)}): {conditioned}")

        evidence = {}
        for node in conditioned:
            state = self.query_state if self.query_state is not None else random.randint(0, 1)
            evidence[str(node)] = state

        try:
            q = infer.query(variables=[root_node], evidence=evidence, show_progress=False)
            return torch.tensor(q.values, dtype=torch.float), evidence
        except Exception as e:
            if self.verbose:
                print(f"Inference failed (intermediate): {e}")
            return torch.tensor([0.5, 0.5], dtype=torch.float), evidence

    def infer_root_given_leaf(self, model, json_data):
        """Perform inference from leaf to root nodes"""
        infer = VariableElimination(model)
        root_node = str(json_data["node_types"]["roots"][0])
        leaf_nodes = json_data["node_types"]["leaves"]

        conditioned_leaves = random.sample(
            leaf_nodes, min(len(leaf_nodes), self.num_leaf_to_infer)
        )

        if self.verbose:
            print(f"Selected conditioned leaf nodes (count={len(conditioned_leaves)}): {conditioned_leaves}")

        evidence = {}
        for leaf in conditioned_leaves:
            state = self.query_state if self.query_state is not None else random.randint(0, 1)
            evidence[str(leaf)] = state

        try:
            q = infer.query(variables=[root_node], evidence=evidence, show_progress=False)
            return torch.tensor(q.values, dtype=torch.float), evidence
        except Exception as e:
            if self.verbose:
                print(f"Inference failed (leaf): {e}")
            return torch.tensor([0.5, 0.5], dtype=torch.float), evidence

class GraphPreprocessor:
    """Handles graph preprocessing with different masking strategies"""

    def __init__(self, mode="distribution", json_folder="graphs",
                 mask_strategy="root_only", query_state=None, verbose=False, 
                 use_intermediate=False, num_leaf_to_infer=2, num_intermediate_to_infer=2):
        self.mode = mode.lower()
        self.mask_strategy = mask_strategy.lower()
        self.json_folder = json_folder
        self.query_state = query_state
        self.verbose = verbose
        self.bn_builder = BayesianNetworkBuilder(verbose=verbose)
        self.use_intermediate = use_intermediate

        # UPDATED: Use separate leaf and intermediate counts
        self.num_leaf_to_infer = int(num_leaf_to_infer)
        self.num_intermediate_to_infer = int(num_intermediate_to_infer)

        # Pass both to InferenceEngine
        self.inference_engine = InferenceEngine(
            num_leaf_to_infer=self.num_leaf_to_infer,
            num_intermediate_to_infer=self.num_intermediate_to_infer,
            query_state=query_state,
            verbose=verbose
        )

    def preprocess_graph(self, data, global_cpd_len, graph_idx=None):
        """Preprocess a single graph with consistent indexing"""
        x = data.x.clone()

        # Updated feature indices
        node_type_idx = 0
        num_parents_idx = 8
        evidence_flag_idx = 9
        cpd_start_idx = 10

        # Insert evidence flag column at index 9 if missing
        if x.shape[1] == 9 + global_cpd_len:
            x = torch.cat([
                x[:, :9],
                torch.zeros(x.size(0), 1),
                x[:, 9:]
            ], dim=1)
            if self.verbose:
                print(f"Graph {graph_idx}: Added evidence_flag column at index {evidence_flag_idx}")

        # Identify node types
        node_types = x[:, node_type_idx]
        root_nodes = (node_types == 0).nonzero(as_tuple=True)[0]
        leaf_nodes = (node_types == 2).nonzero(as_tuple=True)[0]
        intermediate_nodes = (node_types == 1).nonzero(as_tuple=True)[0]

        if len(root_nodes) == 0:
            raise ValueError("No root node found in graph")
        root_node = root_nodes[0].item()
        variable_card = int(x[root_node, 7].item())

        # Generate targets based on mode
        if self.mode in ["distribution", "root_probability"]:
            json_path = os.path.join(self.json_folder, f"detailed_graph_{graph_idx}.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")

            model, json_data = self.bn_builder.build_from_json(json_path)

            # UPDATED: Always use mixed evidence selection when use_intermediate=True
            if self.use_intermediate:
                prob, evidence = self.inference_engine.infer_root_given_mixed_evidence(
                    model, json_data, use_intermediate=True
                )
            else:
                # Use leaf-only inference
                prob, evidence = self.inference_engine.infer_root_given_leaf(model, json_data)

            # Set target based on mode
            if self.mode == "distribution":
                data.y = prob
            else:  # root_probability
                data.y = torch.tensor([prob[0].item()], dtype=torch.float)

            # Store evidence information
            data.evidence_ids = torch.tensor([int(k) for k in evidence.keys()])
            data.evidence_vals = torch.tensor([v for v in evidence.values()])

            # Mark evidence nodes in feature vector
            for eid in data.evidence_ids:
                x[eid, evidence_flag_idx] = 1.0

        elif self.mode == "regression":
            data.y = x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len].clone()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.verbose:
            print(f"Graph {graph_idx} - Before masking root CPD: {x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len]}")

        # ===============================================
        # New Feature Computation: CPD Entropy, Evidence Strength, Distance to Evidence
        # ===============================================

        entropies = []
        for i in range(x.size(0)):
            cpd_vals = x[i, cpd_start_idx:cpd_start_idx + global_cpd_len].cpu().numpy()
            cpd_sum = np.sum(cpd_vals)
            if cpd_sum > 0:
                p = cpd_vals / cpd_sum
            else:
                p = np.ones_like(cpd_vals) / len(cpd_vals)
            entropy = 0 if np.allclose(p, 0) else scipy.stats.entropy(p)
            entropies.append(entropy)
        cpd_entropy = torch.tensor(entropies, dtype=torch.float).unsqueeze(1)

        evidence_strength = x[:, evidence_flag_idx].unsqueeze(1).clone()

        edge_list = data.edge_index.t().cpu().numpy().tolist()
        G = nx.Graph()
        G.add_edges_from(edge_list)
        evidence_nodes = (x[:, evidence_flag_idx] == 1).nonzero(as_tuple=True)[0].tolist()
        num_nodes = x.size(0)
        if len(evidence_nodes) == 0:
            distance_list = [num_nodes] * num_nodes
        else:
            distance_list = []
            for i in range(num_nodes):
                try:
                    dist_values = [nx.shortest_path_length(G, i, int(e)) for e in evidence_nodes if nx.has_path(G, i, int(e))]
                    d = min(dist_values) if len(dist_values) > 0 else num_nodes
                except Exception:
                    d = num_nodes
                distance_list.append(d)
        distance_to_evidence = torch.tensor(distance_list, dtype=torch.float).unsqueeze(1)

        x = torch.cat([x, cpd_entropy, evidence_strength, distance_to_evidence], dim=1)
        data.x = x

        # ===============================================
        # Masking strategy
        # ===============================================

        if self.mask_strategy == "root_only":
            x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len] = 1.0 / variable_card

        elif self.mask_strategy == "evidence_only":
            if hasattr(data, 'evidence_ids'):
                for eid in data.evidence_ids:
                    x[eid, cpd_start_idx:cpd_start_idx + global_cpd_len] = 1.0 / variable_card

        elif self.mask_strategy == "both":
            x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len] = 1.0 / variable_card
            if hasattr(data, 'evidence_ids'):
                for eid in data.evidence_ids:
                    x[eid, cpd_start_idx:cpd_start_idx + global_cpd_len] = 1.0 / variable_card

        elif self.mask_strategy == "none":
            pass

        else:
            raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")

        if self.verbose:
            print(f"Graph {graph_idx} - After masking (strategy: {self.mask_strategy})")
            print(f"  Root CPD: {x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len]}")
            print(f"  Evidence flags: {x[:, evidence_flag_idx].nonzero().flatten()}")
            print(f"  Evidence count: {(x[:, evidence_flag_idx] == 1).sum().item()}")
            print(f"  Target: {data.y}")

        data.root_node = root_node
        data.leaf_nodes = leaf_nodes
        data.intermediate_nodes = intermediate_nodes
        data.mode = self.mode
        data.mask_strategy = self.mask_strategy

        return data


class DataPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Set random seeds
        random.seed(self.config.get("random_seed", 42))
        torch.manual_seed(self.config.get("random_seed", 42))

        # Initialize paths and settings
        self.json_folder = self.config.get("json_folder", self.config.get("output_dir", "generated_graphs"))
        self.output_dir = self.config.get("inference_output_dir", "inference_results")
        self.dataset_path = self.config.get("dataset_path")
        self.global_cpd_len_path = self.config.get("global_cpd_len_path")
        self.mode = self.config.get("mode", "regression")
        self.mask_strategy = self.config.get("mask_strategy", "root_only")
        self.query_state = self.config.get("query_state")
        self.verbose = self.config.get("verbose", False)
        self.use_intermediate = self.config.get("use_intermediate", False)
        
        # UPDATED: Separate leaf and intermediate counts
        self.num_leaf_to_infer = self.config.get("num_leaf_to_infer", 2)
        self.num_intermediate_to_infer = self.config.get("num_intermediate_to_infer", 2)

        self.use_kfold = self.config.get("use_kfold", False)
        self.k_folds = self.config.get("k_folds", 5)
        self.stratify_folds = self.config.get("stratify_folds", True)
        self.fold_random_seed = self.config.get("fold_random_seed", 42)

        if self.verbose:
            print(f"Pipeline Configuration:")
            print(f"  JSON folder: {self.json_folder}")
            print(f"  Mode: {self.mode}")
            print(f"  Mask strategy: {self.mask_strategy}")
            print(f"  Use intermediate: {self.use_intermediate}")
            print(f"  Num leaf to infer: {self.num_leaf_to_infer}")
            print(f"  Num intermediate to infer: {self.num_intermediate_to_infer}")

        os.makedirs(self.output_dir, exist_ok=True)

        self.bn_builder = BayesianNetworkBuilder(verbose=self.verbose)
        self.inference_engine = InferenceEngine(
            num_leaf_to_infer=self.num_leaf_to_infer,
            num_intermediate_to_infer=self.num_intermediate_to_infer,
            verbose=self.verbose
        )
        self.preprocessor = GraphPreprocessor(
            mode=self.mode,
            mask_strategy=self.mask_strategy,
            json_folder=self.json_folder,
            query_state=self.query_state,
            verbose=self.verbose,
            use_intermediate=self.use_intermediate,
            num_leaf_to_infer=self.num_leaf_to_infer,
            num_intermediate_to_infer=self.num_intermediate_to_infer
        )
        
    def create_stratified_kfold_splits(self, processed_data):
        """Create stratified k-fold splits based on target probabilities"""
        print(f"Creating {self.k_folds}-fold cross-validation splits...")

        # Extract targets for stratification
        targets = []
        for graph in processed_data:
            if self.mode == "root_probability":
                targets.append(graph.y.item())
            elif self.mode == "distribution":
                targets.append(graph.y[1].item())  # P(class=1)
            else:  # regression
                targets.append(graph.y.mean().item())

        targets = np.array(targets)

        if self.stratify_folds:
            # Create probability bins for stratification
            discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
            y_binned = discretizer.fit_transform(targets.reshape(-1, 1)).ravel()
            print(f"Stratification bins: {np.bincount(y_binned.astype(int))}")
        else:
            y_binned = np.zeros(len(targets))  # No stratification

        # Create fold splits
        skf = StratifiedKFold(
            n_splits=self.k_folds,
            shuffle=True,
            random_state=self.fold_random_seed
        )

        fold_splits = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.arange(len(processed_data)), y_binned)):
            # Further split train_val into train and val
            train_val_targets = targets[train_val_idx]
            train_val_binned = y_binned[train_val_idx] if self.stratify_folds else np.zeros(len(train_val_idx))

            # Use smaller split for validation within each fold
            inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=self.fold_random_seed + fold_idx)
            train_sub_idx, val_sub_idx = next(inner_skf.split(train_val_idx, train_val_binned))

            train_idx = train_val_idx[train_sub_idx]
            val_idx = train_val_idx[val_sub_idx]

            fold_splits.append({
                'fold': fold_idx,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
                'train_data': [processed_data[i] for i in train_idx],
                'val_data': [processed_data[i] for i in val_idx],
                'test_data': [processed_data[i] for i in test_idx]
            })

            print(f"Fold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

        return fold_splits

    def validate_json_files(self):
        """Validate that JSON files exist"""
        if not os.path.exists(self.json_folder):
            raise FileNotFoundError(f"JSON folder not found: {self.json_folder}")

        json_files = [f for f in os.listdir(self.json_folder) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files in {self.json_folder}")

        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON files found in {self.json_folder}")

        detailed_files = [f for f in json_files if f.startswith('detailed_graph_')]
        print(f"Found {len(detailed_files)} detailed_graph_*.json files")

        return detailed_files

    def run_preprocessing_and_split(self):
        """Run preprocessing and perform K-Fold splits and dataset saving."""

        try:
            dataset = torch.load(self.dataset_path, weights_only=False)
            with open(self.global_cpd_len_path, "r") as f:
                global_cpd_len = int(f.read().strip())
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

        print(f"Loaded dataset with {len(dataset)} graphs.")

        processed = []
        failed_indices = []
        inference_results = []

        # Preprocess all graphs
        for i, graph in enumerate(tqdm(dataset, desc="Processing graphs")):
            try:
                processed_graph = self.preprocessor.preprocess_graph(graph, global_cpd_len, i)
                processed.append(processed_graph)

                # Collect inference results for analysis
                if self.mode in ["distribution", "root_probability"]:
                    inference_results.append({
                        "graph_idx": i,
                        "prob": processed_graph.y.tolist() if hasattr(processed_graph, "y") else None,
                        "evidence": {
                            str(eid.item()): int(eval_.item())
                            for eid, eval_ in zip(
                                getattr(processed_graph, "evidence_ids", []),
                                getattr(processed_graph, "evidence_vals", [])
                            )
                        } if hasattr(processed_graph, "evidence_ids") else None,
                        "mask_strategy": self.mask_strategy
                    })

            except Exception as e:
                if self.verbose:
                    print(f"Error processing graph {i}: {e}")
                failed_indices.append(i)

        if not processed:
            raise RuntimeError("No graphs were successfully processed!")
        print(f"Successfully processed {len(processed)}/{len(dataset)} graphs")
        if failed_indices:
            print(f"Failed to process {len(failed_indices)} graphs")

        # Save inference results if necessary
        if self.mode in ["distribution", "root_probability"]:
            inference_results_path = os.path.join(self.output_dir, "inference_results.json")
            try:
                with open(inference_results_path, "w") as f:
                    json.dump(inference_results, f, indent=2)
                print(f"Inference results saved to {inference_results_path}")
            except Exception as e:
                print(f"Failed to save inference results: {e}")

        # =========================
        # K-FOLD SPLIT LOGIC BELOW
        # =========================
        use_kfold = self.config.get("use_kfold", True)
        n_folds = self.config.get("k_folds", 5)
        stratify = self.config.get("stratify_folds", True)
        fold_random_seed = self.config.get("fold_random_seed", 42)

        if use_kfold:
            print(f"Performing K-Fold split: k={n_folds}, stratify={stratify}")
            n_total = len(processed)

            # -------- Bin targets for stratification --------
            if self.mode == "root_probability":
                y_targets = np.array([float(g.y.item()) for g in processed])
            elif self.mode == "distribution":
                y_targets = np.array([float(g.y[1].item()) for g in processed])
            else:
                y_targets = np.zeros(n_total)

            if stratify:
                discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
                bins = discretizer.fit_transform(y_targets.reshape(-1, 1)).ravel()
            else:
                bins = np.zeros(n_total)

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fold_random_seed)
            fold_data = []

            for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.arange(n_total), bins)):
                # Within fold: further split train_val into train and val
                train_val_targets = y_targets[train_val_idx]
                train_val_bins = bins[train_val_idx] if stratify else np.zeros(len(train_val_idx))
                val_split = 0.2
                val_count = int(len(train_val_idx) * val_split)

                # Shuffle train_val indices for validation split
                rng = np.random.RandomState(fold_random_seed + fold_idx)
                permuted = rng.permutation(train_val_idx)
                val_idx = permuted[:val_count]
                train_idx = permuted[val_count:]
                # Collect graph objects
                train_graphs = [processed[i] for i in train_idx]
                val_graphs = [processed[i] for i in val_idx]
                test_graphs = [processed[i] for i in test_idx]

                fold_data.append((train_graphs, val_graphs, test_graphs))
                # Save to disk
                os.makedirs("datasets/folds", exist_ok=True)
                torch.save(train_graphs, f"datasets/folds/fold_{fold_idx}_train.pt")
                torch.save(val_graphs, f"datasets/folds/fold_{fold_idx}_val.pt")
                torch.save(test_graphs, f"datasets/folds/fold_{fold_idx}_test.pt")
                print(f"Fold {fold_idx + 1}: Saved train/val/test graphs [{len(train_graphs)}, {len(val_graphs)}, {len(test_graphs)}]")

            # Save fold metadata
            fold_metadata = {
                "k_folds": n_folds,
                "stratified": stratify,
                "fold_sizes": [(len(tr), len(val), len(te)) for tr, val, te in fold_data]
            }
            with open("datasets/folds/fold_metadata.json", "w") as f:
                json.dump(fold_metadata, f, indent=2)
            print("\nAll K-Fold splits generated and saved.")
        else:
            # === Classic single split ===
            random.shuffle(processed)
            n = len(processed)
            train_ratio = self.config.get("train_split", 0.7)
            val_ratio = self.config.get("val_split", 0.2)
            train_end = int(train_ratio * n)
            val_end = int((train_ratio + val_ratio) * n)
            train = processed[:train_end]
            val = processed[train_end:val_end]
            test = processed[val_end:]
            os.makedirs("datasets", exist_ok=True)
            torch.save(train, "datasets/train.pt")
            torch.save(val, "datasets/val.pt")
            torch.save(test, "datasets/test.pt")
            print(f"Classic split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    def run_pipeline(self):
        """Run the complete pipeline"""
        print("Starting data preprocessing pipeline...")
        try:
            self.run_preprocessing_and_split()
            print("Pipeline completed successfully!")
        except Exception as e:
            print(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run_pipeline()
