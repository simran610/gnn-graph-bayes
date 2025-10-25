import os
import json
import random
import torch
import yaml
import numpy as np
import scipy.stats
import networkx as nx
import glob
from pathlib import Path
from typing import Dict, List

from pgmpy.readwrite import BIFReader
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from torch_geometric.data import Data


class BIFDownloader:
    BNLEARN_REPO = "https://www.bnlearn.com/bnrepository"
    EXAMPLE_NETWORKS = [
        "asia", "cancer", "child", "insurance", "alarm",
        "barley", "hailfinder", "hepar2", "win95pts"
    ]

    def __init__(self, output_dir="dataset_bif_files", verbose=False):
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)

    def download_bif(self, network_name: str) -> str:
        bif_url = f"{self.BNLEARN_REPO}/{network_name}/{network_name}.bif"
        local_path = os.path.join(self.output_dir, f"{network_name}.bif")
        if os.path.exists(local_path):
            if self.verbose:
                print(f"✓ {network_name}.bif already exists")
            return local_path
        try:
            if self.verbose:
                print(f"Downloading {network_name}.bif...")
            from urllib.request import urlretrieve
            urlretrieve(bif_url, local_path)
            if self.verbose:
                print(f"✓ Saved to {local_path}")
            return local_path
        except Exception as e:
            print(f"✗ Failed to download {network_name}: {e}")
            return None

    def download_multiple(self, networks=None):
        if networks is None:
            networks = self.EXAMPLE_NETWORKS[:3]
        paths = []
        for net in networks:
            path = self.download_bif(net)
            if path:
                paths.append(path)
        return paths


class BIFToGraphConverter:
    def __init__(self, verbose=False):
        self.verbose = verbose

    @staticmethod
    def convert_np_types(obj):
        if isinstance(obj, dict):
            return {k: BIFToGraphConverter.convert_np_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [BIFToGraphConverter.convert_np_types(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def load_bif(self, bif_path: str):
        try:
            reader = BIFReader(bif_path)
            model = reader.get_model()
            metadata = {
                "nodes": list(model.nodes()),
                "edges": list(model.edges()),
                "num_nodes": model.number_of_nodes(),
                "num_edges": model.number_of_edges()
            }
            if self.verbose:
                print(f"✓ Loaded BIF: {metadata['num_nodes']} nodes, {metadata['num_edges']} edges")
            return model, metadata
        except Exception as e:
            print(f"✗ Error loading BIF: {e}")
            raise

    def get_node_types(self, model: BayesianNetwork):
        roots, intermediates, leaves = [], [], []
        for node in model.nodes():
            parents = list(model.predecessors(node))
            children = list(model.successors(node))
            if len(parents) == 0 and len(children) > 0:
                roots.append(node)
            elif len(parents) > 0 and len(children) > 0:
                intermediates.append(node)
            else:
                leaves.append(node)
        return {"roots": roots, "intermediates": intermediates, "leaves": leaves}

    def extract_cpd_as_dict(self, model, node: str):
        cpd = model.get_cpds(node)
        if cpd is None:
            return {"variable_card": 2, "evidence": [], "evidence_card": [], "values": [0.5, 0.5]}
        return {
            "variable_card": cpd.variable_card,
            "evidence": list(cpd.variables[1:]) if len(cpd.variables) > 1 else [],
            "evidence_card": list(cpd.cardinality[1:]) if len(cpd.cardinality) > 1 else [],
            "values": cpd.values.flatten().tolist()
        }

    def convert_to_json(self, model, bif_name: str, output_dir: str = "bif_json"):
        os.makedirs(output_dir, exist_ok=True)
        node_types = self.get_node_types(model)
        nodes_dict = {}
        for node in model.nodes():
            cpd = self.extract_cpd_as_dict(model, node)
            nodes_dict[str(node)] = {
                "cpd": cpd,
                "node_type": self._get_node_type_label(node, node_types)
            }
        edges = [{"source": str(src), "target": str(tgt)} for src, tgt in model.edges()]
        json_data = {
            "network_name": bif_name,
            "nodes": nodes_dict,
            "edges": edges,
            "node_types": {
                "roots": [str(n) for n in node_types["roots"]],
                "intermediates": [str(n) for n in node_types["intermediates"]],
                "leaves": [str(n) for n in node_types["leaves"]],
            },
        }
        json_data = BIFToGraphConverter.convert_np_types(json_data)
        json_path = os.path.join(output_dir, f"{bif_name}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        if self.verbose:
            print(f"✓ Saved JSON: {json_path}")
        return json_path

    def _get_node_type_label(self, node: str, node_types: Dict):
        if node in node_types["roots"]:
            return 0
        elif node in node_types["intermediates"]:
            return 1
        else:
            return 2


class GraphFeatureExtractor:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def compute_structural_features(self, G: nx.DiGraph, node: str):
        in_deg = G.in_degree[node]
        out_deg = G.out_degree[node]
        try:
            betweenness = nx.betweenness_centrality(G).get(node, 0.0)
            closeness = nx.closeness_centrality(G).get(node, 0.0)
            pagerank = nx.pagerank(G).get(node, 0.0)
            degree_cent = nx.degree_centrality(G).get(node, 0.0)
        except Exception:
            betweenness = closeness = pagerank = degree_cent = 0.0
        return [float(in_deg), float(out_deg), float(betweenness), float(closeness), float(pagerank), float(degree_cent)]

    def extract_node_features(self, model, json_data, G, node, node_type: int, global_cpd_len: int):
        cpd_info = json_data["nodes"][node]["cpd"]
        variable_card = cpd_info["variable_card"]
        num_parents = len(cpd_info["evidence"])
        struct_feat = self.compute_structural_features(G, node)
        cpd_values = cpd_info["values"]
        if len(cpd_values) < global_cpd_len:
            cpd_values = cpd_values + [0.0] * (global_cpd_len - len(cpd_values))
        else:
            cpd_values = cpd_values[:global_cpd_len]
        features = [float(node_type)] + struct_feat + [float(variable_card), float(num_parents), 0.0] + cpd_values
        return np.array(features, dtype=np.float32)

    def create_graph_dataset(self, model, json_data, global_cpd_len):
        G = nx.DiGraph()
        nodes = list(json_data["nodes"].keys())
        G.add_nodes_from(nodes)
        for edge in json_data["edges"]:
            G.add_edge(edge["source"], edge["target"])
        node_to_idx = {n: i for i, n in enumerate(sorted(nodes))}
        node_features = [
            self.extract_node_features(model, json_data, G, node,
                                       json_data["nodes"][node]["node_type"], global_cpd_len)
            for node in sorted(nodes)
        ]
        x = torch.tensor(np.array(node_features), dtype=torch.float32)
        edge_list = [[node_to_idx[edge["source"]], node_to_idx[edge["target"]]] for edge in json_data["edges"]]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([0.5, 0.5], dtype=torch.float32))
        return data, node_to_idx


class EnhancedGraphPreprocessor:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def compute_cpd_entropy(self, cpd_values):
        p = cpd_values / np.sum(cpd_values) if np.sum(cpd_values) > 0 else np.ones_like(cpd_values) / len(cpd_values)
        return 0. if np.allclose(p, 0) else scipy.stats.entropy(p)

    def compute_distance_to_evidence(self, edge_index, evidence_indices, num_nodes):
        if len(evidence_indices) == 0:
            return np.full(num_nodes, num_nodes, dtype=np.float32)
        G = nx.Graph()
        G.add_edges_from(edge_index.t().cpu().numpy())
        dists = []
        for i in range(num_nodes):
            min_dist = num_nodes
            for e in evidence_indices:
                try:
                    if nx.has_path(G, i, int(e)):
                        dist = nx.shortest_path_length(G, i, int(e))
                        if dist < min_dist:
                            min_dist = dist
                except:
                    pass
            dists.append(min_dist)
        return np.array(dists, dtype=np.float32)

    def add_missing_features(self, data, global_cpd_len, graph_idx=None):
        x = data.x.clone().float()
        e_flag = 9
        cpd_start = 10
        cpd_end = cpd_start + global_cpd_len
        num_nodes = x.size(0)
        entropies = [self.compute_cpd_entropy(x[i, cpd_start:cpd_end].cpu().numpy()) for i in range(num_nodes)]
        cpd_entropy = torch.tensor(entropies, dtype=torch.float32).unsqueeze(1)
        evidence_str = x[:, e_flag:e_flag + 1].clone()
        evidence_indices = (x[:, e_flag] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        distances = torch.tensor(self.compute_distance_to_evidence(data.edge_index, evidence_indices, num_nodes),
                                 dtype=torch.float32).unsqueeze(1)
        x_enh = torch.cat([x, cpd_entropy, evidence_str, distances], dim=1)
        data.x = x_enh
        if self.verbose and graph_idx is not None:
            print(f"  Graph {graph_idx}: Features {x.shape[1]} → {x_enh.shape[1]}")
        return data


class BIFInferenceEngine:
    def __init__(self, num_leaf_to_infer=2, num_intermediate_to_infer=2, query_state=None, verbose=True):
        self.num_leaf_to_infer = num_leaf_to_infer
        self.num_intermediate_to_infer = num_intermediate_to_infer
        self.query_state = query_state
        self.verbose = verbose

    def infer(self, model, json_data, use_intermediate=True):
        infer = VariableElimination(model)
        root_node = str(json_data["node_types"]["roots"][0])
        evidence = {}
        selected_nodes = []

        # Enforce query_state for evidence nodes if set
        if use_intermediate:
            intermediates = json_data.get("node_types", {}).get("intermediates", [])
            if intermediates:
                chosen = random.sample(intermediates, min(self.num_intermediate_to_infer, len(intermediates)))
                selected_nodes.extend(chosen)
                for node in chosen:
                    assigned_state = self.query_state if self.query_state is not None else 0
                    evidence[str(node)] = assigned_state

        leaves = json_data.get("node_types", {}).get("leaves", [])
        if leaves:
            chosen = random.sample(leaves, min(self.num_leaf_to_infer, len(leaves)))
            selected_nodes.extend(chosen)
            for node in chosen:
                assigned_state = self.query_state if self.query_state is not None else 0
                evidence[str(node)] = assigned_state

        if self.verbose:
            print(f"DEBUG: Inference on root node '{root_node}'")
            print(f"DEBUG: Evidence nodes ({len(selected_nodes)}): {selected_nodes}")
            print(f"DEBUG: Evidence assignments: {evidence}")

        try:
            query_result = infer.query(variables=[root_node], evidence=evidence, show_progress=False)
            probs = query_result.values
            if self.verbose:
                print(f"DEBUG: Inference probabilities: {probs}")
            if abs(probs.sum() - 1.0) > 1e-5:
                print("WARNING: Inference probabilities do not sum up to 1.")
            probs_tensor = torch.tensor(probs, dtype=torch.float)
            return probs_tensor, evidence
        except Exception as e:
            print(f"ERROR: Inference failed with exception: {e}")
            fallback = torch.tensor([0.5, 0.5], dtype=torch.float)
            return fallback, evidence


class EndToEndPipeline:
    def __init__(self, config_path="config.yaml", verbose=True):
        self.verbose = verbose
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.dataset_path = self.config.get("dataset_path", "datasets")
        self.global_cpd_len_path = self.config.get("global_cpd_len_path", "global_datasets/global_cpd_len.txt")
        self.inference_output_dir = self.config.get("inference_output_dir", "saved_inference_outputs")
        self.dataset_dir = self.dataset_path
        self.cpd_len_dir = os.path.dirname(self.global_cpd_len_path) or "."
        self.bif_dir = "dataset_bif_files"
        self.json_dir = "bif_json"
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.cpd_len_dir, exist_ok=True)
        os.makedirs(self.inference_output_dir, exist_ok=True)
        os.makedirs(self.bif_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

    def run(self, networks=None, train_split=None, val_split=None):
        if train_split is None:
            train_split = self.config.get("train_split", 0.7)
        if val_split is None:
            val_split = self.config.get("val_split", 0.2)
        print("=" * 60)
        print("BIF → GNN DATASET PIPELINE")
        print("=" * 60)
        print("\n[1/5] Downloading BIF files...")
        downloader = BIFDownloader(self.bif_dir, verbose=self.verbose)
        bif_paths = downloader.download_multiple(networks)
        print(f"✓ Downloaded {len(bif_paths)} BIF files")
        print("\n[2/5] Converting BIF to JSON...")
        converter = BIFToGraphConverter(verbose=self.verbose)
        json_paths = []
        for bif_path in bif_paths:
            name = Path(bif_path).stem
            model, _ = converter.load_bif(bif_path)
            json_path = converter.convert_to_json(model, name, self.json_dir)
            json_paths.append(json_path)
        print(f"✓ Created {len(json_paths)} JSON files")
        print("\n[3/5] Creating PyG graphs...")

        # Determine max CPD length globally
        all_cpd_lens = []
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            for n in json_data["nodes"]:
                all_cpd_lens.append(len(json_data["nodes"][n]["cpd"]["values"]))
        global_cpd_len = max(all_cpd_lens) if all_cpd_lens else 0

        graphs = []
        inference_results = []
        inference_engine = BIFInferenceEngine(
            verbose=self.verbose,
            query_state=self.config.get("query_state", 0))  # Enforce query state from config

        for idx, json_path in enumerate(json_paths):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            model, _ = converter.load_bif(os.path.join(self.bif_dir, f"{json_data['network_name']}.bif"))
            extractor = GraphFeatureExtractor(verbose=self.verbose)
            data, _ = extractor.create_graph_dataset(model, json_data, global_cpd_len=global_cpd_len)
            prob, evidence = inference_engine.infer(model, json_data, use_intermediate=True)
            data.y = torch.tensor([prob[0].item()], dtype=torch.float)  # use P(root=0)
            graphs.append(data)
            evidence_dict = {str(k): int(v) for k, v in evidence.items()}
            inference_results.append({
                "graph_idx": idx,
                "prob": [prob[0].item()],
                "evidence": evidence_dict,
                "mask_strategy": "evidence_only"
            })
        print(f"✓ Created {len(graphs)} graphs")
        print(f"  Global CPD length: {global_cpd_len}")
        print("\n[4/5] Adding enhanced features...")
        enhancer = EnhancedGraphPreprocessor(verbose=self.verbose)
        enhanced_graphs = []
        for idx, graph in enumerate(graphs):
            enhanced = enhancer.add_missing_features(graph, global_cpd_len, graph_idx=idx)
            enhanced_graphs.append(enhanced)
        print(f"✓ Enhanced {len(enhanced_graphs)} graphs")
        print("\n[5/5] Splitting and saving dataset...")
        random.shuffle(enhanced_graphs)
        n = len(enhanced_graphs)
        train_end = int(train_split * n)
        val_end = int((train_split + val_split) * n)
        train_data = enhanced_graphs[:train_end]
        val_data = enhanced_graphs[train_end:val_end]
        test_data = enhanced_graphs[val_end:]
        train_path = os.path.join(self.dataset_dir, "train.pt")
        val_path = os.path.join(self.dataset_dir, "val.pt")
        test_path = os.path.join(self.dataset_dir, "test.pt")
        torch.save(train_data, train_path)
        torch.save(val_data, val_path)
        torch.save(test_data, test_path)
        with open(self.global_cpd_len_path, 'w') as f:
            f.write(str(int(global_cpd_len)))
        metadata = {
            "total_graphs": len(enhanced_graphs),
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
            "global_cpd_len": int(global_cpd_len),
            "total_features": enhanced_graphs[0].x.shape[1] if enhanced_graphs else 0,
            "feature_names": [
                "node_type", "in_degree", "out_degree", "betweenness", "closeness",
                "pagerank", "degree_centrality", "variable_card", "num_parents",
                "evidence_flag", f"cpd_0...cpd_{global_cpd_len - 1}",
                "cpd_entropy", "evidence_strength", "distance_to_evidence"
            ]
        }
        metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        inference_path = os.path.join(self.inference_output_dir, "inference_results.json")
        with open(inference_path, "w") as f:
            json.dump(inference_results, f, indent=2)
        print(f"✓ Saved datasets:")
        print(f"  Train: {train_path} ({len(train_data)} graphs)")
        print(f"  Val: {val_path} ({len(val_data)} graphs)")
        print(f"  Test: {test_path} ({len(test_data)} graphs)")
        print(f"  Global CPD Length: {self.global_cpd_len_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Inference Results saved to: {inference_path}")
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        return train_data, val_data, test_data, metadata


if __name__ == "__main__":
    pipeline = EndToEndPipeline(config_path="config.yaml", verbose=True)
    bif_files = glob.glob(os.path.join("dataset_bif_files", "*.bif"))
    networks = [os.path.splitext(os.path.basename(f))[0] for f in bif_files]
    train, val, test, metadata = pipeline.run(networks=networks)
    print("\nDataset ready for GNN training!")
    print(f"Sample graph shape: {train[0].x.shape}")
    print(f"Sample graph edges: {train[0].edge_index.shape}")
    print(f"\nDataset Summary:")
    print(f"  Total graphs: {metadata['total_graphs']}")
    print(f"  Train: {metadata['train']} graphs")
    print(f"  Val: {metadata['val']} graphs")
    print(f"  Test: {metadata['test']} graphs")
    print(f"  Global CPD Length: {metadata['global_cpd_len']}")
    print(f"  Total Features per node: {metadata['total_features']}")
