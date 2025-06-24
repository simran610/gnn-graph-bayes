# data_preprocessor.py

import os
import json
import random
import torch
import yaml
import numpy as np
from tqdm import tqdm
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class BayesianNetworkBuilder:
    """Builds Bayesian networks from JSON with error handling"""
    
    def __init__(self, normalize_cpds=True):
        self.normalize_cpds = normalize_cpds
    
    def build_from_json(self, filepath):
        """Build pgmpy BN from JSON file"""
        with open(filepath, "r") as f:
            json_data = json.load(f)

        model = DiscreteBayesianNetwork()
        
        # Add edges
        for edge in json_data["edges"]:
            model.add_edge(str(edge["source"]), str(edge["target"]))

        # Add CPDs
        for node_id, node_info in json_data["nodes"].items():
            try:
                cpd = self._create_cpd(node_id, node_info)
                model.add_cpds(cpd)
            except Exception:
                
                cpd = self._create_uniform_cpd(node_id, node_info)
                model.add_cpds(cpd)

        model.check_model()
        return model, json_data
    
    def _create_uniform_cpd(self, node_id, node_info):
        """Create uniform CPD as fallback"""
        cpd_info = node_info["cpd"]
        variable_card = cpd_info["variable_card"]
        evidence = [str(e) for e in cpd_info.get("evidence", [])]
        evidence_card = cpd_info.get("evidence_card", [])
        
        uniform_prob = 1.0 / variable_card
        
        if evidence:
            num_cols = np.prod(evidence_card)
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
            num_cols = np.prod(evidence_card)
            expected_length = variable_card * num_cols
            
            if len(values) < expected_length:
                values.extend([0.0] * (expected_length - len(values)))
         
            values_matrix = []
            for var_state in range(variable_card):
                row = []
                for parent_config in range(num_cols):
                    idx = var_state * num_cols + parent_config
                    row.append(values[idx] if idx < len(values) else 0.0)
                values_matrix.append(row)
        else:
            values_matrix = [[values[i]] for i in range(min(variable_card, len(values)))]
            while len(values_matrix) < variable_card:
                values_matrix.append([0.0])

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
                
                if col_sum == 0:
                    normalized_value = 1.0 / variable_card
                else:
                    normalized_value = values_matrix[var_state][col] / col_sum
                
                normalized_row.append(normalized_value)
            normalized_matrix.append(normalized_row)
        
        return normalized_matrix


class InferenceEngine:
    """Handles probabilistic inference"""
    
    def __init__(self, num_leaf_to_infer=2):
        self.num_leaf_to_infer = num_leaf_to_infer
    
    def infer_root_given_leaf(self, model, json_data):
        """Perform inference from leaf to root nodes"""
        infer = VariableElimination(model)
        root_node = str(json_data["node_types"]["roots"][0])
        leaf_nodes = json_data["node_types"]["leaves"]

        conditioned_leaves = random.sample(
            leaf_nodes, min(len(leaf_nodes), self.num_leaf_to_infer)
        )
        
        # Generate evidence
        evidence = {}
        for leaf in conditioned_leaves:
            card = json_data["nodes"][str(leaf)]["cpd"]["variable_card"]
            state = random.randint(0, card - 1)
            evidence[str(leaf)] = state

        # Perform inference
        q = infer.query(variables=[root_node], evidence=evidence, show_progress=False)
        return torch.tensor(q.values, dtype=torch.float), evidence


class GraphPreprocessor:
    """Handles graph preprocessing"""
    
    def __init__(self, mode="regression", json_folder="generated_graphs"):
        self.mode = mode
        self.json_folder = json_folder
        self.bn_builder = BayesianNetworkBuilder()
        self.inference_engine = InferenceEngine()
    
    def preprocess_graph(self, data, global_cpd_len, graph_idx=None):
        """Preprocess a single graph"""
        x = data.x.clone()
        node_type_idx = 0
        cpd_start_idx = 9

        # Find root and leaf nodes
        node_types = x[:, node_type_idx]
        root_indices = (node_types == 0).nonzero(as_tuple=False).squeeze()
        leaf_indices = (node_types == 2).nonzero(as_tuple=False).squeeze()

        # Validate nodes exist
        if root_indices.numel() == 0:
            raise ValueError("Root node missing")
        if leaf_indices.numel() == 0:
            raise ValueError("Leaf node missing")

        # Handle tensor dimensions
        root_node = root_indices.item() if root_indices.dim() == 0 else root_indices[0].item()
        if leaf_indices.dim() == 0:
            leaf_indices = leaf_indices.unsqueeze(0)

        if self.mode == "regression":
            return self._preprocess_regression(data, x, root_node, leaf_indices, 
                                             cpd_start_idx, global_cpd_len)
        elif self.mode == "conditional_probability":
            return self._preprocess_conditional(data, x, root_node, leaf_indices, 
                                              cpd_start_idx, global_cpd_len, graph_idx)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _preprocess_regression(self, data, x, root_node, leaf_indices, 
                             cpd_start_idx, global_cpd_len):
        """Preprocess for regression mode"""
        y = x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len].clone()
        x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len] = 0.0
        
        data.x = x
        data.y = y
        data.root_node = root_node
        data.leaf_nodes = leaf_indices
        return data

    def _preprocess_conditional(self, data, x, root_node, leaf_indices, 
                              cpd_start_idx, global_cpd_len, graph_idx):
        """Preprocess for conditional probability mode"""
        json_filename = f"detailed_graph_{graph_idx}.json"
        json_path = os.path.join(self.json_folder, json_filename)
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        model, json_data = self.bn_builder.build_from_json(json_path)
        prob, evidence = self.inference_engine.infer_root_given_leaf(model, json_data)

        # Mask the CPDs
        x[root_node, cpd_start_idx:cpd_start_idx + global_cpd_len] = 0.0
        for leaf_idx in leaf_indices:
            leaf_node = leaf_idx.item() if leaf_idx.dim() == 0 else leaf_idx
            x[leaf_node, cpd_start_idx:cpd_start_idx + global_cpd_len] = 0.0

        data.x = x
        data.y = prob
        data.root_node = root_node
        data.conditioned_leaf_nodes = torch.tensor(list(map(int, evidence.keys())), dtype=torch.long)
        data.evidence = evidence
        return data


class DataPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Set random seeds
        random.seed(self.config.get("random_seed", 42))
        torch.manual_seed(self.config.get("random_seed", 42))

        # Initialize components
        self.json_folder = self.config.get("output_dir", "generated_graphs")
        self.output_dir = self.config.get("inference_output_dir", "inference_results")
        self.dataset_path = self.config.get("dataset_path")
        self.global_cpd_len_path = self.config.get("global_cpd_len_path")
        self.mode = self.config.get("mode", "regression")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize processors
        self.bn_builder = BayesianNetworkBuilder()
        self.inference_engine = InferenceEngine(self.config.get("num_leaf_to_infer", 2))
        self.preprocessor = GraphPreprocessor(self.mode, self.json_folder)

    def run_preprocessing_and_split(self):
        """Run preprocessing and dataset splitting"""
        dataset = torch.load(self.dataset_path, weights_only=False)
        with open(self.global_cpd_len_path, "r") as f:
            global_cpd_len = int(f.read().strip())

        print(f"Loaded dataset with {len(dataset)} graphs")
        
        # Preprocess graphs
        processed_dataset = []
        failed = 0
        
        for i, graph in enumerate(tqdm(dataset, desc="Preprocessing graphs")):
            try:
                processed_graph = self.preprocessor.preprocess_graph(
                    graph, global_cpd_len, graph_idx=i
                )
                processed_dataset.append(processed_graph)
            except Exception as e:
                failed += 1

        print(f"Successfully preprocessed {len(processed_dataset)}/{len(dataset)} graphs")
        
        if not processed_dataset:
            raise RuntimeError("No graphs were successfully preprocessed!")

        # Split dataset
        random.shuffle(processed_dataset)
        total = len(processed_dataset)
        train_len = int(self.config.get("train_split", 0.8) * total)
        val_len = int(self.config.get("val_split", 0.1) * total)

        train_set = processed_dataset[:train_len]
        val_set = processed_dataset[train_len:train_len + val_len]
        test_set = processed_dataset[train_len + val_len:]

        # Save splits
        os.makedirs("saved_datasets", exist_ok=True)
        torch.save(train_set, "saved_datasets/train.pt")
        torch.save(val_set, "saved_datasets/val.pt")
        torch.save(test_set, "saved_datasets/test.pt")

        print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

  
    def run_preprocessing_and_split(self):
        """Run preprocessing and dataset splitting with better error handling"""
        dataset = torch.load(self.dataset_path, weights_only=False)
        with open(self.global_cpd_len_path, "r") as f:
            global_cpd_len = int(f.read().strip())

        print(f"Loaded dataset with {len(dataset)} graphs")
        
        # Preprocess graphs
        processed_dataset = []
        failed = 0
        
        for i, graph in enumerate(tqdm(dataset, desc="Preprocessing graphs")):
            try:
                processed_graph = self.preprocessor.preprocess_graph(
                    graph, global_cpd_len, graph_idx=i
                )
                processed_dataset.append(processed_graph)
            except Exception as e:
                failed += 1

        print(f"Successfully preprocessed {len(processed_dataset)}/{len(dataset)} graphs")
        
        if not processed_dataset:
            raise RuntimeError("No graphs were successfully preprocessed!")

        # Split dataset
        random.shuffle(processed_dataset)
        total = len(processed_dataset)
        train_len = int(self.config.get("train_split", 0.8) * total)
        val_len = int(self.config.get("val_split", 0.1) * total)

        train_set = processed_dataset[:train_len]
        val_set = processed_dataset[train_len:train_len + val_len]
        test_set = processed_dataset[train_len + val_len:]
        save_dir = "saved_datasets"
    
        if os.path.exists(save_dir):
            import shutil
            shutil.rmtree(save_dir)
        
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            print("Saving train set...")
            torch.save(train_set, os.path.join(save_dir, "train.pt"))
            print("✓ Train set saved successfully")
            
            print("Saving validation set...")
            torch.save(val_set, os.path.join(save_dir, "val.pt"))
            print("✓ Validation set saved successfully")
            
            print("Saving test set...")
            torch.save(test_set, os.path.join(save_dir, "test.pt"))
            print("✓ Test set saved successfully")
            
        except Exception as e:
            print(f"Error saving datasets: {e}")
            
           
            print("Trying to save to current directory...")
            try:
                torch.save(train_set, "train_dataset.pt")
                torch.save(val_set, "val_dataset.pt") 
                torch.save(test_set, "test_dataset.pt")
                print("✓ Datasets saved to current directory")
            except Exception as e2:
                print(f"Failed to save to current directory: {e2}")
                
                print("\nDiagnostic information:")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Directory exists: {os.path.exists(save_dir)}")
                print(f"Directory writable: {os.access(save_dir, os.W_OK)}")
                print(f"Train set size: {len(train_set)} items")
                print(f"Val set size: {len(val_set)} items")
                print(f"Test set size: {len(test_set)} items")
                
                raise e2

        print(f"Dataset split completed:")
        print(f"  Train: {len(train_set)} graphs")
        print(f"  Validation: {len(val_set)} graphs") 
        print(f"  Test: {len(test_set)} graphs")

    def run_inference(self):
        """Run inference on JSON files"""
        results = []
        json_files = sorted([f for f in os.listdir(self.json_folder) if f.endswith(".json")])

        for file in tqdm(json_files, desc="Processing graphs"):
            json_path = os.path.join(self.json_folder, file)
            try:
                model, json_data = self.bn_builder.build_from_json(json_path)
                prob, evidence = self.inference_engine.infer_root_given_leaf(model, json_data)
                
                results.append({
                    "filename": file, 
                    "prob": prob.tolist(), 
                    "evidence": evidence
                })

                torch.save(prob, os.path.join(self.output_dir, f"{file.replace('.json', '')}_y.pt"))
                
            except Exception as e:
                print(f"Error in {file}: {e}")

        # Save all results
        with open(os.path.join(self.output_dir, "all_inference_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    def run_pipeline(self):
        """Run the pipeline"""
        print("Starting pipeline...")
        
        if self.mode == "conditional_probability":
            print("Running inference...")
            self.run_inference()
        
        print("Preprocessing and splitting dataset...")
        self.run_preprocessing_and_split()
        
        print("Pipeline complete!")


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run_pipeline()