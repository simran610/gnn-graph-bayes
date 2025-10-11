import os
import json
import time
import random
import numpy as np
import yaml
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ------------- Configuration --------------
JSON_DIR = "./generated_graphs"
CONFIG_PATH = "./config.yaml"
OUTPUT_CSV = "inference_results.csv"
ROOT_NODE = "0"

class BNInferencePipeline:
    def __init__(self, config_path, json_dir, root_node="0"):
        self.config = self.load_config(config_path)
        self.json_dir = json_dir
        self.root_node = root_node
        self.results = []
        
        # Set random seeds
        random.seed(self.config.get("random_seed", 42))
        np.random.seed(self.config.get("random_seed", 42))
        
        # Load max CPD length
        try:
            with open("global_datasets/global_cpd_len.txt") as f:
                self.max_cpd_len = int(f.read().strip())
        except:
            print("Warning: Could not load global_cpd_len.txt, using default")
            self.max_cpd_len = 10000
    
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg
    
    def build_bn_from_json(self, js):
        """Build Bayesian Network from JSON structure with proper 2D array handling"""
        edges = []
        for e in js.get('edges', []):
            edges.append((str(e['source']), str(e['target'])))
        
        model = BayesianNetwork(edges)
        
        for node_idx, node in js.get("nodes", {}).items():
            cpd = node.get("cpd", None)
            if cpd is not None:
                variable = str(node_idx)
                variable_card = cpd["variable_card"]
                values = np.array(cpd["values"])
                
                evidence = [str(ev) for ev in cpd.get("evidence", [])] if cpd.get("evidence") else None
                evidence_card = cpd.get("evidence_card", None) if evidence else None
                
                # CRITICAL FIX: Properly reshape values for pgmpy
                if evidence and evidence_card:
                    # For nodes with parents: reshape to (variable_card, product of evidence_card)
                    num_parent_states = int(np.prod(evidence_card))
                    expected_len = variable_card * num_parent_states
                    values = values[:expected_len]  # Trim to expected length
                    
                    try:
                        # pgmpy wants (variable_card, num_parent_states)
                        values = values.reshape(variable_card, num_parent_states)
                    except:
                        # Fallback: create uniform distribution
                        values = np.ones((variable_card, num_parent_states)) / variable_card
                else:
                    # For root nodes: reshape to (variable_card, 1)
                    values = values[:variable_card]
                    values = values.reshape(variable_card, 1)
                
                # Ensure values sum to 1 along axis 0 for each parent configuration
                col_sums = values.sum(axis=0)
                col_sums[col_sums == 0] = 1  # Avoid division by zero
                values = values / col_sums
                
                model.add_cpds(
                    TabularCPD(variable, variable_card, values,
                              evidence=evidence, evidence_card=evidence_card))
        
        model.check_model()
        return model
    
    def infer_probability(self, model, evidence, query, state):
        """Run inference on Bayesian Network"""
        infer = VariableElimination(model)
        result = infer.query([query], evidence=evidence, show_progress=False)
        return result.values[state]
    
    def select_evidence_nodes(self, js):
        """Select random evidence nodes from intermediates and leaves"""
        num_leaf = self.config.get("num_leaf_to_infer", 4)
        num_intermediate = self.config.get("num_evidence_to_infer", 4)
        
        node_types = js.get('node_types', {})
        intermediates = [str(n) for n in node_types.get('intermediates', [])]
        leaves = [str(n) for n in node_types.get('leaves', [])]
        
        chosen_intermediate = random.sample(
            intermediates, min(len(intermediates), num_intermediate)
        ) if intermediates else []
        
        chosen_leaf = random.sample(
            leaves, min(len(leaves), num_leaf)
        ) if leaves else []
        
        evidence_nodes = chosen_intermediate + chosen_leaf
        evidence_states = [random.choice([0, 1]) for _ in evidence_nodes]
        evidence_dict = dict(zip(evidence_nodes, evidence_states))
        
        return evidence_dict, len(chosen_intermediate), len(chosen_leaf)
    
    def process_single_graph(self, filepath, fname):
        """Process a single graph file"""
        start_time = time.time()
        query_state = self.config.get("query_state", 0)
        
        try:
            with open(filepath, 'r') as f:
                js = json.load(f)
        except Exception as e:
            return None, f"JSON load error: {e}"
        
        # Build Bayesian Network
        try:
            model = self.build_bn_from_json(js)
        except Exception as e:
            return None, f"Model build error: {e}"
        
        # Select evidence nodes
        evidence_dict, inter_count, leaf_count = self.select_evidence_nodes(js)
        
        # Get ground truth probability
        try:
            true_prob = js["nodes"][self.root_node]["cpd"]["values"][query_state]
        except:
            return None, "Root node or state missing"
        
        # Run inference
        try:
            est_prob = self.infer_probability(model, evidence_dict, self.root_node, query_state)
            inf_time = time.time() - start_time
            inference_success = True
        except Exception as e:
            est_prob = np.nan
            inf_time = time.time() - start_time
            inference_success = False
        
        # Calculate metrics
        mae = abs(est_prob - true_prob) if not np.isnan(est_prob) else np.nan
        rmse = np.sqrt(mae**2) if not np.isnan(mae) else np.nan
        accuracy = 1 if not np.isnan(mae) and mae < 0.1 else 0
        
        result = {
            "filename": fname,
            "estimated_prob": est_prob,
            "true_prob": true_prob,
            "mae": mae,
            "rmse": rmse,
            "accuracy": accuracy,
            "inference_time": inf_time,
            "intermediate_evidence_count": inter_count,
            "leaf_evidence_count": leaf_count,
            "inference_success": inference_success
        }
        
        return result, None
    
    def run(self):
        """Run the full pipeline"""
        print("="*80)
        print("PGMPY BAYESIAN NETWORK INFERENCE PIPELINE")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"JSON Directory: {self.json_dir}")
        print(f"Config: {CONFIG_PATH}")
        print("-"*80)
        
        # Get all JSON files
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith(".json")]
        total_files = len(json_files)
        print(f"Found {total_files} JSON graph files")
        print("-"*80)
        
        # Start timing
        pipeline_start = time.time()
        
        # Process each graph
        skipped = 0
        for idx, fname in enumerate(json_files, start=1):
            filepath = os.path.join(self.json_dir, fname)
            result, error = self.process_single_graph(filepath, fname)
            
            if result is None:
                if idx <= 5 or idx % 5000 == 0:  # Show first 5 errors and every 5000th
                    print(f"[{idx}/{total_files}] SKIP {fname}: {error}")
                skipped += 1
                continue
            
            self.results.append(result)
            
            # Progress update
            if idx % 1000 == 0 or idx == total_files:
                success_count = len(self.results)
                print(f"[{idx}/{total_files}] Processed | Success: {success_count} | Skipped: {skipped}")
        
        # Calculate total time
        total_time = time.time() - pipeline_start
        
        # Generate and print summary statistics
        self.print_summary(total_files, skipped, total_time)
        
        # Save results
        if self.results:
            self.save_results()
        else:
            print("\n⚠ WARNING: No successful inferences. No CSV file generated.")
        
        return self.results
    
    def print_summary(self, total_files, skipped, total_time):
        """Print comprehensive summary statistics"""
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        # Overall Statistics
        print(f"\n{'OVERALL STATISTICS':<40}")
        print("-"*80)
        print(f"{'Total graphs:':<40} {total_files}")
        print(f"{'Successfully processed:':<40} {len(self.results)}")
        print(f"{'Skipped (errors):':<40} {skipped}")
        
        if len(self.results) == 0:
            print("\n⚠ ERROR: All graphs were skipped. No successful inferences.")
            print("Please check your JSON file format and CPD structure.")
            print("="*80)
            return
        
        df = pd.DataFrame(self.results)
        successful = df['inference_success'].sum()
        
        print(f"{'Successful inferences:':<40} {successful}")
        print(f"{'Failed inferences:':<40} {len(self.results) - successful}")
        print(f"{'Success rate:':<40} {successful/len(self.results)*100:.2f}%")
        
        # Timing Statistics
        print(f"\n{'TIMING STATISTICS':<40}")
        print("-"*80)
        print(f"{'Total pipeline time:':<40} {total_time:.4f} seconds")
        print(f"{'Average time per graph:':<40} {total_time/len(self.results):.4f} seconds")
        print(f"{'Throughput:':<40} {len(self.results)/total_time:.2f} graphs/second")
        
        if successful > 0:
            valid_df = df[df['inference_success'] == True]
            print(f"{'Average inference time (success):':<40} {valid_df['inference_time'].mean():.4f} seconds")
            print(f"{'Min inference time:':<40} {valid_df['inference_time'].min():.4f} seconds")
            print(f"{'Max inference time:':<40} {valid_df['inference_time'].max():.4f} seconds")
            print(f"{'Median inference time:':<40} {valid_df['inference_time'].median():.4f} seconds")
        
        # Accuracy Metrics
        if successful > 0:
            valid_df = df[df['inference_success'] == True]
            print(f"\n{'ACCURACY METRICS (SUCCESSFUL INFERENCES)':<40}")
            print("-"*80)
            print(f"{'Mean Absolute Error (MAE):':<40} {valid_df['mae'].mean():.6f}")
            print(f"{'Root Mean Square Error (RMSE):':<40} {valid_df['rmse'].mean():.6f}")
            print(f"{'Accuracy (MAE < 0.1):':<40} {valid_df['accuracy'].mean()*100:.2f}%")
            print(f"{'MAE Std Dev:':<40} {valid_df['mae'].std():.6f}")
            print(f"{'MAE Min:':<40} {valid_df['mae'].min():.6f}")
            print(f"{'MAE Max:':<40} {valid_df['mae'].max():.6f}")
            print(f"{'MAE Median:':<40} {valid_df['mae'].median():.6f}")
            
            # Additional percentiles
            print(f"{'MAE 25th percentile:':<40} {valid_df['mae'].quantile(0.25):.6f}")
            print(f"{'MAE 75th percentile:':<40} {valid_df['mae'].quantile(0.75):.6f}")
            print(f"{'MAE 95th percentile:':<40} {valid_df['mae'].quantile(0.95):.6f}")
        
        # Evidence Statistics
        print(f"\n{'EVIDENCE NODE STATISTICS':<40}")
        print("-"*80)
        print(f"{'Avg intermediate evidence nodes:':<40} {df['intermediate_evidence_count'].mean():.2f}")
        print(f"{'Avg leaf evidence nodes:':<40} {df['leaf_evidence_count'].mean():.2f}")
        print(f"{'Total avg evidence nodes:':<40} {(df['intermediate_evidence_count'] + df['leaf_evidence_count']).mean():.2f}")
        
        print("\n" + "="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
    
    def save_results(self):
        """Save results to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✓ Detailed results saved to: {OUTPUT_CSV}")


def main():
    """Main entry point"""
    pipeline = BNInferencePipeline(
        config_path=CONFIG_PATH,
        json_dir=JSON_DIR,
        root_node=ROOT_NODE
    )
    
    results = pipeline.run()
    
    if len(results) > 0:
        print(f"\n✓ Pipeline completed successfully with {len(results)} inferences")
    else:
        print(f"\n✗ Pipeline failed - all graphs were skipped")
        print("Debug: Check one of your JSON files to verify CPD structure")
    
    return results


if __name__ == "__main__":
    main()