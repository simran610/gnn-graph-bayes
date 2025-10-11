"""
End-to-end pipeline: BIF file → JSON → NetworkX → PyG graphs → Enhanced features
Downloads BIF files, converts to graphs, adds features, saves dataset ready for GNN training
"""

import os
import json
import random
import torch
import yaml
import numpy as np
import scipy.stats
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve
import warnings

from pgmpy.readwrite import BIFReader
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from torch_geometric.data import Data, InMemoryDataset


class BIFDownloader:
    """Download BIF files from bnlearn repository"""
    
    BNLEARN_REPO = "https://www.bnlearn.com/bnrepository"
    
    EXAMPLE_NETWORKS = [
        "asia", "cancer", "child", "insurance", "alarm",
        "barley", "hailfinder", "hepar2", "win95pts"
    ]
    
    def __init__(self, output_dir="bif_files", verbose=False):
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)
    
    def download_bif(self, network_name: str) -> str:
        """Download a BIF file and return local path"""
        bif_url = f"{self.BNLEARN_REPO}/{network_name}.bif"
        local_path = os.path.join(self.output_dir, f"{network_name}.bif")
        
        if os.path.exists(local_path):
            if self.verbose:
                print(f"✓ {network_name}.bif already exists")
            return local_path
        
        try:
            if self.verbose:
                print(f"Downloading {network_name}.bif...")
            urlretrieve(bif_url, local_path)
            if self.verbose:
                print(f"✓ Saved to {local_path}")
            return local_path
        except Exception as e:
            print(f"✗ Failed to download {network_name}: {e}")
            return None
    
    def download_multiple(self, networks: List[str] = None) -> List[str]:
        """Download multiple BIF files"""
        if networks is None:
            networks = self.EXAMPLE_NETWORKS[:3]  # Download first 3 by default
        
        paths = []
        for net in networks:
            path = self.download_bif(net)
            if path:
                paths.append(path)
        
        return paths


class BIFToGraphConverter:
    """Convert BIF files to PyG graphs with networkx intermediate"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def load_bif(self, bif_path: str) -> Tuple[BayesianNetwork, Dict]:
        """Load BIF file and create BayesianNetwork model"""
        try:
            reader = BIFReader(bif_path)
            model = reader.get_model()
            
            # Create metadata
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
    
    def get_node_types(self, model: BayesianNetwork) -> Dict[str, List]:
        """Classify nodes as root, intermediate, or leaf"""
        roots = []
        intermediates = []
        leaves = []
        
        for node in model.nodes():
            parents = list(model.predecessors(node))
            children = list(model.successors(node))
            
            if len(parents) == 0 and len(children) > 0:
                roots.append(node)
            elif len(parents) > 0 and len(children) > 0:
                intermediates.append(node)
            else:  # leaf or isolated
                leaves.append(node)
        
        return {
            "roots": roots,
            "intermediates": intermediates,
            "leaves": leaves
        }
    
    def extract_cpd_as_dict(self, model: BayesianNetwork, node: str) -> Dict:
        """Extract CPD for a node as dictionary"""
        cpd = model.get_cpds(node)
        
        if cpd is None:
            return {
                "variable_card": 2,
                "evidence": [],
                "evidence_card": [],
                "values": [0.5, 0.5]
            }
        
        cpd_dict = {
            "variable_card": cpd.variable_card,
            "evidence": list(cpd.variables[1:]) if len(cpd.variables) > 1 else [],
            "evidence_card": list(cpd.cardinality[1:]) if len(cpd.cardinality) > 1 else [],
            "values": cpd.values.flatten().tolist()
        }
        
        return cpd_dict
    
    def convert_to_json(self, model: BayesianNetwork, bif_name: str, 
                       output_dir: str = "bif_json") -> str:
        """Convert BayesianNetwork to JSON format"""
        os.makedirs(output_dir, exist_ok=True)
        
        node_types = self.get_node_types(model)
        
        # Build node dictionary with CPDs
        nodes_dict = {}
        for node in model.nodes():
            cpd = self.extract_cpd_as_dict(model, node)
            nodes_dict[str(node)] = {
                "cpd": cpd,
                "node_type": self._get_node_type_label(node, node_types)
            }
        
        # Build edges
        edges = [{"source": str(src), "target": str(tgt)} for src, tgt in model.edges()]
        
        # Complete JSON structure
        json_data = {
            "network_name": bif_name,
            "nodes": nodes_dict,
            "edges": edges,
            "node_types": {
                "roots": [str(n) for n in node_types["roots"]],
                "intermediates": [str(n) for n in node_types["intermediates"]],
                "leaves": [str(n) for n in node_types["leaves"]]
            }
        }
        
        json_path = os.path.join(output_dir, f"{bif_name}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        
        if self.verbose:
            print(f"✓ Saved JSON: {json_path}")
        
        return json_path
    
    def _get_node_type_label(self, node: str, node_types: Dict) -> int:
        """Map node to type label: 0=root, 1=intermediate, 2=leaf"""
        if node in node_types["roots"]:
            return 0
        elif node in node_types["intermediates"]:
            return 1
        else:
            return 2
    
    def bif_to_nx_graph(self, json_data: Dict) -> nx.DiGraph:
        """Convert JSON representation to NetworkX DiGraph"""
        G = nx.DiGraph()
        
        # Add nodes
        for node_id in json_data["nodes"].keys():
            G.add_node(str(node_id))
        
        # Add edges
        for edge in json_data["edges"]:
            G.add_edge(str(edge["source"]), str(edge["target"]))
        
        return G


class GraphFeatureExtractor:
    """Extract features for GNN nodes"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def compute_structural_features(self, G: nx.DiGraph, node: str) -> List[float]:
        """Compute structural graph features for a node"""
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        
        # Centrality measures
        try:
            betweenness = nx.betweenness_centrality(G).get(node, 0.0)
            closeness = nx.closeness_centrality(G).get(node, 0.0)
            pagerank = nx.pagerank(G).get(node, 0.0)
            degree_cent = nx.degree_centrality(G).get(node, 0.0)
        except Exception:
            betweenness = closeness = pagerank = degree_cent = 0.0
        
        return [
            float(in_deg),
            float(out_deg),
            float(betweenness),
            float(closeness),
            float(pagerank),
            float(degree_cent)
        ]
    
    def extract_node_features(self, model: BayesianNetwork, json_data: Dict,
                             G: nx.DiGraph, node: str, node_type: int) -> np.ndarray:
        """Extract all features for a node"""
        
        cpd_info = json_data["nodes"][node]["cpd"]
        variable_card = cpd_info["variable_card"]
        num_parents = len(cpd_info["evidence"])
        
        # Structural features: [in_degree, out_degree, betweenness, closeness, pagerank, degree_centrality]
        struct_feat = self.compute_structural_features(G, node)
        
        # Node features: [node_type, in_degree, out_degree, betweenness, closeness, pagerank, degree_centrality, 
        #                 variable_card, num_parents, evidence_flag]
        features = [
            float(node_type),
            struct_feat[0],  # in_degree
            struct_feat[1],  # out_degree
            struct_feat[2],  # betweenness
            struct_feat[3],  # closeness
            struct_feat[4],  # pagerank
            struct_feat[5],  # degree_centrality
            float(variable_card),
            float(num_parents),
            0.0  # evidence_flag (set during preprocessing)
        ]
        
        # Append CPD values
        cpd_values = cpd_info["values"]
        features.extend(cpd_values)
        
        return np.array(features, dtype=np.float32)
    
    def create_graph_dataset(self, model: BayesianNetwork, json_data: Dict) -> Data:
        """Create a PyG Data object from BayesianNetwork"""
        
        G = nx.DiGraph()
        for src, tgt in json_data["edges"]:
            G.add_edge(src, tgt)
        
        nodes = list(json_data["nodes"].keys())
        node_to_idx = {node: idx for idx, node in enumerate(sorted(nodes))}
        
        # Extract node features
        node_features = []
        for node in sorted(nodes):
            node_type = json_data["nodes"][node]["node_type"]
            features = self.extract_node_features(model, json_data, G, node, node_type)
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Build edge index
        edge_list = []
        for src, tgt in json_data["edges"]:
            edge_list.append([node_to_idx[src], node_to_idx[tgt]])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([0.5, 0.5], dtype=torch.float32)
        )
        
        return data, node_to_idx


class EnhancedGraphPreprocessor:
    """Add CPD entropy, evidence strength, and distance to evidence features"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def compute_cpd_entropy(self, cpd_values):
        """Compute Shannon entropy of CPD values"""
        cpd_sum = np.sum(cpd_values)
        if cpd_sum > 0:
            p = cpd_values / cpd_sum
        else:
            p = np.ones_like(cpd_values) / len(cpd_values)
        
        if np.allclose(p, 0):
            entropy = 0.0
        else:
            entropy = scipy.stats.entropy(p)
        
        return entropy
    
    def compute_distance_to_evidence(self, edge_index, evidence_indices, num_nodes):
        """Compute shortest path distance from each node to nearest evidence node"""
        
        if len(evidence_indices) == 0:
            return np.full(num_nodes, num_nodes, dtype=np.float32)
        
        edge_list = edge_index.t().cpu().numpy().tolist()
        G = nx.Graph()
        G.add_edges_from(edge_list)
        
        distances = []
        for i in range(num_nodes):
            min_dist = num_nodes
            for ev_idx in evidence_indices:
                try:
                    if nx.has_path(G, i, int(ev_idx)):
                        dist = nx.shortest_path_length(G, i, int(ev_idx))
                        min_dist = min(min_dist, dist)
                except Exception:
                    pass
            distances.append(min_dist)
        
        return np.array(distances, dtype=np.float32)
    
    def add_missing_features(self, data, global_cpd_len, graph_idx=None):
        """Add CPD entropy, evidence strength, and distance to evidence"""
        
        x = data.x.clone().float()
        
        evidence_flag_idx = 9
        cpd_start_idx = 10
        cpd_end_idx = cpd_start_idx + global_cpd_len
        
        num_nodes = x.size(0)
        
        # Compute CPD Entropy
        cpd_entropies = []
        for i in range(num_nodes):
            cpd_vals = x[i, cpd_start_idx:cpd_end_idx].cpu().numpy()
            entropy = self.compute_cpd_entropy(cpd_vals)
            cpd_entropies.append(entropy)
        
        cpd_entropy_feat = torch.tensor(cpd_entropies, dtype=torch.float32).unsqueeze(1)
        
        # Evidence Strength
        evidence_strength_feat = x[:, evidence_flag_idx:evidence_flag_idx+1].clone()
        
        # Distance to Evidence
        evidence_indices = (x[:, evidence_flag_idx] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        distances = self.compute_distance_to_evidence(data.edge_index, evidence_indices, num_nodes)
        distance_feat = torch.tensor(distances, dtype=torch.float32).unsqueeze(1)
        
        # Concatenate new features
        x_enhanced = torch.cat([x, cpd_entropy_feat, evidence_strength_feat, distance_feat], dim=1)
        data.x = x_enhanced
        
        if self.verbose and graph_idx is not None:
            print(f"  Graph {graph_idx}: Features {x.shape[1]} → {x_enhanced.shape[1]}")
        
        return data


class EndToEndPipeline:
    """Complete pipeline: BIF → JSON → Graphs → Enhanced Features → Dataset"""
    
    def __init__(self, config_path="config.yaml", verbose=True):
        self.verbose = verbose
        
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Extract paths from config
        self.dataset_path = self.config.get("dataset_path", "data_processing/dataset.pt")
        self.global_cpd_len_path = self.config.get("global_cpd_len_path", "global_datasets/global_cpd_len.txt")
        
        # Create directories
        self.dataset_dir = os.path.dirname(self.dataset_path) or "."
        self.cpd_len_dir = os.path.dirname(self.global_cpd_len_path) or "."
        self.bif_dir = "bif_files"
        self.json_dir = "bif_json"
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.cpd_len_dir, exist_ok=True)
        os.makedirs(self.bif_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
    
    def run(self, networks: List[str] = None, train_split=None, val_split=None):
        """Run complete pipeline"""
        
        # Use config values if not provided
        if train_split is None:
            train_split = self.config.get("train_split", 0.7)
        if val_split is None:
            val_split = self.config.get("val_split", 0.2)
        
        print("="*60)
        print("BIF → GNN DATASET PIPELINE")
        print("="*60)
        
        # Step 1: Download BIF files
        print("\n[1/5] Downloading BIF files...")
        downloader = BIFDownloader(self.bif_dir, verbose=self.verbose)
        bif_paths = downloader.download_multiple(networks)
        print(f"✓ Downloaded {len(bif_paths)} BIF files")
        
        # Step 2: Convert to JSON
        print("\n[2/5] Converting BIF to JSON...")
        converter = BIFToGraphConverter(verbose=self.verbose)
        json_paths = []
        
        for bif_path in bif_paths:
            name = Path(bif_path).stem
            model, _ = converter.load_bif(bif_path)
            json_path = converter.convert_to_json(model, name, self.json_dir)
            json_paths.append(json_path)
        
        print(f"✓ Created {len(json_paths)} JSON files")
        
        # Step 3: Create PyG graphs
        print("\n[3/5] Creating PyG graphs...")
        graphs = []
        global_cpd_len = 0
        
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            model, _ = converter.load_bif(
                os.path.join(self.bif_dir, f"{json_data['network_name']}.bif")
            )
            
            extractor = GraphFeatureExtractor(verbose=self.verbose)
            data, _ = extractor.create_graph_dataset(model, json_data)
            graphs.append(data)
            
            # Track max CPD length
            cpd_len = data.x.shape[1] - 10
            global_cpd_len = max(global_cpd_len, cpd_len)
        
        print(f"✓ Created {len(graphs)} graphs")
        print(f"  Global CPD length: {global_cpd_len}")
        
        # Step 4: Add enhanced features
        print("\n[4/5] Adding enhanced features...")
        enhancer = EnhancedGraphPreprocessor(verbose=self.verbose)
        enhanced_graphs = []
        
        for idx, graph in enumerate(graphs):
            enhanced = enhancer.add_missing_features(graph, global_cpd_len, graph_idx=idx)
            enhanced_graphs.append(enhanced)
        
        print(f"✓ Enhanced {len(enhanced_graphs)} graphs")
        
        # Step 5: Split and save
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
        
        # Save global CPD length
        with open(self.global_cpd_len_path, 'w') as f:
            f.write(str(int(global_cpd_len)))
        
        # Save metadata
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
                "evidence_flag", f"cpd_0...cpd_{global_cpd_len-1}",
                "cpd_entropy", "evidence_strength", "distance_to_evidence"
            ]
        }
        
        metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved datasets:")
        print(f"  Train: {train_path} ({len(train_data)} graphs)")
        print(f"  Val:   {val_path} ({len(val_data)} graphs)")
        print(f"  Test:  {test_path} ({len(test_data)} graphs)")
        print(f"  Global CPD Length: {self.global_cpd_len_path}")
        print(f"  Metadata: {metadata_path}")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        
        return train_data, val_data, test_data, metadata


if __name__ == "__main__":
    # Run pipeline
    pipeline = EndToEndPipeline(config_path="config.yaml", verbose=True)
    
    # Download and process these networks
    networks = ["asia", "cancer", "alarm"]
    
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