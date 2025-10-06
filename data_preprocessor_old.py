# data_preprocessor.py

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
10+    | CPD values          | Flattened Conditional Probability Table values

Important:
- Evidence flag is inserted at index 9 during preprocessing and shifts CPD start index to 10.
- Normalization applies to indices 2-6.
- Masking applies to CPD feature columns starting at index 10.

Note: Comments created using chatGPT-4, based on the provided context and code structure.
"""


import os
import json
import random
import torch
import yaml
import numpy as np
from tqdm import tqdm
# from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


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
    """Handles probabilistic inference"""
    
    def __init__(self, num_evidence_to_infer=2, query_state=None, verbose=False):
        self.num_evidence_to_infer = num_evidence_to_infer
        self.query_state = query_state  # None for random, or specify 0/1
        self.verbose = verbose

    def infer_root_given_intermediate(self, model, json_data):
        """Perform inference from intermediate to root nodes"""
        infer = VariableElimination(model)
        root_node = str(json_data["node_types"]["roots"][0])
        intermediate_nodes = json_data["node_types"].get("intermediates", [])

        if not intermediate_nodes:
            if self.verbose:
                print("No intermediate nodes found, falling back to leaf-based inference")
            return self.infer_root_given_leaf(model, json_data)

        # Select random intermediate nodes as evidence
        conditioned = random.sample(
            intermediate_nodes, min(len(intermediate_nodes), self.num_evidence_to_infer)
        )

        evidence = {}
        for node in conditioned:
            if self.query_state is not None:
                state = self.query_state
            else:
                state = random.randint(0, 1)
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

        # Select random leaf nodes as evidence
        conditioned_leaves = random.sample(
            leaf_nodes, min(len(leaf_nodes), self.num_evidence_to_infer)
        )
        
        evidence = {}
        for leaf in conditioned_leaves:
            if self.query_state is not None:
                state = self.query_state
            else:
                state = random.randint(0, 1)
            evidence[str(leaf)] = state

        try:
            q = infer.query(variables=[root_node], evidence=evidence, show_progress=False)
            return torch.tensor(q.values, dtype=torch.float), evidence
        except Exception as e:
            if self.verbose:
                print(f"Inference failed: {e}")
            return torch.tensor([0.5, 0.5], dtype=torch.float), evidence


class GraphPreprocessor:
    """Handles graph preprocessing with different masking strategies"""
    
    def __init__(self, mode="distribution", json_folder="graphs", 
                 mask_strategy="root_only", query_state=None, verbose=False, use_intermediate=False):
        self.mode = mode.lower()
        self.mask_strategy = mask_strategy.lower()
        self.json_folder = json_folder
        self.query_state = query_state
        self.verbose = verbose
        self.bn_builder = BayesianNetworkBuilder(verbose=verbose)
        self.inference_engine = InferenceEngine(query_state=query_state, verbose=verbose)
        self.use_intermediate = use_intermediate
    
    def preprocess_graph(self, data, global_cpd_len, graph_idx=None):
        """Preprocess a single graph with consistent indexing"""
        x = data.x.clone()
       
        # Updated feature indices
        node_type_idx = 0
        num_parents_idx = 8
        evidence_flag_idx = 9
        cpd_start_idx = 10  # CPD values start at index 10

        # Insert evidence flag column at index 9 if missing
        if x.shape[1] == 9 + global_cpd_len:  # 9 features before CPDs (without evidence_flag)

            x = torch.cat([
                x[:, :9],                     # Columns 0 to 8
                torch.zeros(x.size(0), 1),   # New evidence_flag column
                x[:, 9:]                     # CPDs
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
            
            # Choose inference strategy
            if self.use_intermediate and len(intermediate_nodes) > 0:
                prob, evidence = self.inference_engine.infer_root_given_intermediate(model, json_data)
            else:
                prob, evidence = self.inference_engine.infer_root_given_leaf(model, json_data)
            
            # Set target based on mode
            if self.mode == "distribution":
                data.y = prob  # [P(0), P(1)]
            else:  # root_probability
                data.y = torch.tensor([prob[0].item()], dtype=torch.float)
                
            # Store evidence information for model to use
            data.evidence_ids = torch.tensor([int(k) for k in evidence.keys()])
            data.evidence_vals = torch.tensor([v for v in evidence.values()])
            
            # IMPORTANT: Mark evidence nodes in feature vector
            for eid in data.evidence_ids:
                x[eid, evidence_flag_idx] = 1.0
                
        elif self.mode == "regression":
            # For regression, predict the CPD values directly
            data.y = x[root_node, cpd_start_idx:cpd_start_idx+global_cpd_len].clone()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        if self.verbose:
            print(f"Graph {graph_idx} - Before masking root CPD: {x[root_node, cpd_start_idx:cpd_start_idx+global_cpd_len]}")

        # # Normalize selected features (indices 2-6) 
        # features_to_normalize = [2, 3, 4, 5, 6]

        # for i in features_to_normalize:
        #     col = x[:, i]
        #     std = col.std()
        #     if std > 0:
        #         mean = col.mean()
        #         x[:, i] = (col - mean) / std

        # Apply masking strategy
        if self.mask_strategy == "root_only":
            # Only mask root node CPD
            #x[root_node, cpd_start_idx:cpd_start_idx+global_cpd_len] = 0.0
            # Instead of masking to 0
            x[root_node, cpd_start_idx:cpd_start_idx+global_cpd_len] = 1.0/variable_card #variable_cardinalities[root_node]  # Example: get cardinality for root node
            
        elif self.mask_strategy == "evidence_only":
            # Only mask evidence nodes CPD (if any)
            if hasattr(data, 'evidence_ids'):
                for eid in data.evidence_ids:
                    #x[eid, cpd_start_idx:cpd_start_idx+global_cpd_len] = 0.0
                    x[eid, cpd_start_idx:cpd_start_idx+global_cpd_len] = 1.0/variable_card
                    
        elif self.mask_strategy == "both":
            # Mask both root and evidence nodes CPD
            #x[root_node, cpd_start_idx:cpd_start_idx+global_cpd_len] = 0.0
            x[root_node, cpd_start_idx:cpd_start_idx+global_cpd_len] = 1.0/variable_card
            if hasattr(data, 'evidence_ids'):
                for eid in data.evidence_ids:
                    #x[eid, cpd_start_idx:cpd_start_idx+global_cpd_len] = 0.0
                    x[eid, cpd_start_idx:cpd_start_idx+global_cpd_len] = 1.0/variable_card
                    
        elif self.mask_strategy == "none":
            # No masking
            pass
        else:
            raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")

        if self.verbose:
            print(f"Graph {graph_idx} - After masking (strategy: {self.mask_strategy})")
            print(f"  Root CPD: {x[root_node, cpd_start_idx:cpd_start_idx+global_cpd_len]}")
            print(f"  Evidence flags: {x[:, evidence_flag_idx].nonzero().flatten()}")
            print(f"  Target: {data.y}")

        # Verify JSON comparison for debugging
        if self.verbose and self.mode in ["distribution", "root_probability"]:
            try:
                json_path = os.path.join(self.json_folder, f"detailed_graph_{graph_idx}.json")
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                root_id = str(int(root_node))
                if root_id in json_data["nodes"]:
                    root_json = json_data["nodes"][root_id]
                    print(f"  JSON root CPD: {root_json['cpd']['values']}")
                    print(f"  Tensor root CPD: {x[root_node, cpd_start_idx:cpd_start_idx+global_cpd_len].tolist()}")
            except Exception as e:
                print(f"  JSON comparison failed: {e}")

        # Store final processed data
        data.x = x
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

        # Set random seeds for reproducibility
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
        
        # Debug information
        if self.verbose:
            print(f"Pipeline Configuration:")
            print(f"  JSON folder: {self.json_folder}")
            print(f"  Mode: {self.mode}")
            print(f"  Mask strategy: {self.mask_strategy}")
            print(f"  Use intermediate: {self.use_intermediate}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize processors
        self.bn_builder = BayesianNetworkBuilder(verbose=self.verbose)
        self.inference_engine = InferenceEngine(
            self.config.get("num_evidence_to_infer", 2), 
            verbose=self.verbose
        )
        self.preprocessor = GraphPreprocessor(
            mode=self.mode,
            mask_strategy=self.mask_strategy,
            json_folder=self.json_folder,
            query_state=self.query_state,
            verbose=self.verbose,
            use_intermediate=self.use_intermediate
        )

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
        """Run preprocessing and dataset splitting"""
        try:
            dataset = torch.load(self.dataset_path, weights_only=False)
            with open(self.global_cpd_len_path, "r") as f:
                global_cpd_len = int(f.read().strip())
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
        
        print(f"Loaded dataset with {len(dataset)} graphs")
        print(f"Global CPD length: {global_cpd_len}")
        
        # Validate JSON files for inference modes
        if self.mode in ["distribution", "root_probability"]:
            try:
                json_files = self.validate_json_files()
            except Exception as e:
                print(f"JSON validation failed: {e}")
                raise
        
        processed = []
        failed_indices = []
        inference_results = []

        for i, graph in enumerate(tqdm(dataset, desc="Processing graphs")):
            try:
                processed_graph = self.preprocessor.preprocess_graph(
                    graph, global_cpd_len, i)
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
            if self.verbose:
                print(f"Failed indices: {failed_indices[:10]}...")

        # Save inference results
        if self.mode in ["distribution", "root_probability"]:
            inference_results_path = os.path.join(self.output_dir, "inference_results.json")
            try:
                with open(inference_results_path, "w") as f:
                    json.dump(inference_results, f, indent=2)
                print(f"Inference results saved to {inference_results_path}")
            except Exception as e:
                print(f"Failed to save inference results: {e}")

        # Split dataset
        random.shuffle(processed)
        n = len(processed)
        train_ratio = self.config.get("split_ratios", {}).get("train", 0.7)
        val_ratio = self.config.get("split_ratios", {}).get("val", 0.2)
        
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)
        
        train = processed[:train_end]
        val = processed[train_end:val_end]
        test = processed[val_end:]

        # Save splits
        os.makedirs("datasets", exist_ok=True)
        try:
            torch.save(train, "datasets/train.pt")
            torch.save(val, "datasets/val.pt")
            torch.save(test, "datasets/test.pt")
            print("Datasets saved successfully")
        except Exception as e:
            raise RuntimeError(f"Error saving datasets: {e}")

        print(f"Dataset split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

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