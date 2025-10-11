import json
import yaml
import os
import numpy as np
import pandas as pd
import time
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, State


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def build_cpt_rows_from_position_map(position_map, vals, var_card):
    """
    Build CPT rows from position_map.
    Each row: [parent_state_1, parent_state_2, ..., child_state, probability]
    """
    cpt_rows = []
    for item in position_map:
        parent_config = item['parent_config']  # list of parent states
        child_state = item['child_state']
        flat_pos = item['flattened_position']
        prob = vals[flat_pos]
        
        # Build row: parent configs + child state + probability
        row = parent_config + [child_state, prob]
        cpt_rows.append(row)
    
    return cpt_rows


def build_bn_from_cpds(graph_json):
    """
    Build Bayesian Network from graph JSON with CPD data.
    Uses recursive DFS to handle any graph structure.
    """
    nodes = graph_json['nodes']
    edges = graph_json.get('edges', [])
    
    # Map node ID to its parents
    node_parents_map = {}
    for edge in edges:
        src = edge['source']
        tgt = edge['target']
        node_parents_map.setdefault(tgt, []).append(src)
    
    # Sort parents to ensure consistent ordering
    for node_id in node_parents_map:
        node_parents_map[node_id].sort()
    
    bn_states = {}
    processing_stack = set()  # Track nodes being processed (detect cycles)
    
    def process_node_recursive(node_id_int):
        """
        Recursively process a node and all its parents first.
        This ensures all parent states exist before creating child CPTs.
        """
        # Already processed
        if node_id_int in bn_states:
            return
        
        # Cycle detection
        if node_id_int in processing_stack:
            raise ValueError(f"Cycle detected involving node {node_id_int}")
        
        processing_stack.add(node_id_int)
        
        if str(node_id_int) not in nodes:
            raise ValueError(f"Node {node_id_int} not found in graph")
        
        node_info = nodes[str(node_id_int)]
        cpd = node_info['cpd']
        vals = cpd['values']
        var_card = cpd['variable_card']
        
        parents = node_parents_map.get(node_id_int, [])
        
        if len(parents) == 0:
            # Root node: simple discrete distribution
            probs = {i: vals[i] for i in range(var_card)}
            dist = DiscreteDistribution(probs)
            bn_states[node_id_int] = State(dist, name=f"node_{node_id_int}")
        else:
            # Non-root node: ensure all parents are processed first
            for parent_id in parents:
                process_node_recursive(parent_id)
            
            # Now create CPT with all parent states available
            position_map = cpd['position_map']
            cpt_rows = build_cpt_rows_from_position_map(position_map, vals, var_card)
            
            parent_states = [bn_states[p_id] for p_id in parents]
            cpt = ConditionalProbabilityTable(cpt_rows, parent_states)
            bn_states[node_id_int] = State(cpt, name=f"node_{node_id_int}")
        
        processing_stack.remove(node_id_int)
    
    # Process all nodes
    for node_id_str in nodes.keys():
        node_id_int = int(node_id_str)
        process_node_recursive(node_id_int)
    
    # Build model
    model = BayesianNetwork("BN from CPDs")
    for state in bn_states.values():
        model.add_state(state)
    
    # Add edges
    for edge in edges:
        parent_id = edge['source']
        child_id = edge['target']
        if parent_id in bn_states and child_id in bn_states:
            model.add_edge(bn_states[parent_id], bn_states[child_id])
    
    model.bake()
    return model, bn_states
    


def infer_root_prob(model, bn_states, root_id, evidence_dict, query_state):
    """
    Calculate P(root_node=query_state | evidence)
    """
    # Build evidence dict with proper naming
    evidence = {f"node_{nid}": val for nid, val in evidence_dict.items()}
    
    # Get beliefs
    beliefs = model.predict_proba(evidence)
    
    # Find root node index and extract probability
    root_state_name = f"node_{root_id}"
    for idx, state in enumerate(model.states):
        if state.name == root_state_name:
            return beliefs[idx].probabilities[query_state]
    
    raise ValueError(f"Root node {root_id} not found in model")


def debug_single_case(graph_json, evidence_dict, root_node_id, query_state):
    """Debug a single case to see what's going wrong"""
    print("\n" + "="*70)
    print("DEBUG: Testing single case")
    print("="*70)
    
    try:
        print(f"Building BN...")
        model, bn_states = build_bn_from_cpds(graph_json)
        print(f"✓ Built BN with {len(bn_states)} nodes")
        
        print(f"\nModel states: {[s.name for s in model.states]}")
        print(f"\nEvidence: {evidence_dict}")
        print(f"Looking for root node: node_{root_node_id}, query_state: {query_state}")
        
        print(f"\nPerforming inference...")
        predicted_prob = infer_root_prob(model, bn_states, root_node_id, 
                                        evidence_dict, query_state)
        print(f"✓ Predicted probability: {predicted_prob}")
        return predicted_prob
        
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}")
        print(f"Message: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    start_time = time.time()
    
    config = load_config("config.yaml")
    
    graph_dir = config.get("output_dir", "./generated_graphs")
    output_dir = config.get("inference_output_dir", "saved_inference_outputs")
    num_graphs = config.get("num_graphs")
    os.makedirs(output_dir, exist_ok=True)
    
    inference_results_path = os.path.join(output_dir, "inference_results.json")
    output_path = os.path.join(output_dir, "pomegranate_predictions.json")
    metrics_path = os.path.join(output_dir, "pomegranate_metrics.json")
    
    root_node_id = config.get("root_node_id", 0)
    query_state = config.get("query_state", 0)
    verbose = config.get("verbose", False)
    random_seed = config.get("random_seed", None)
    
    if random_seed is not None:
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    if not os.path.exists(inference_results_path):
        print(f"Error: {inference_results_path} not found")
        return
    
    inference_results = load_json(inference_results_path)
    print(f"Loaded {len(inference_results)} inference cases")
    
    import glob
    graph_files = sorted(glob.glob(os.path.join(graph_dir, "detailed_graph_*.json")))
    print(f"Found {len(graph_files)} graph files in {graph_dir}")
    
    if num_graphs and len(graph_files) != num_graphs:
        print(f"Warning: Found {len(graph_files)} graphs but config expects {num_graphs}")
    
    print("Loading and caching all graphs...")
    graphs_cache = {}
    for graph_path in graph_files:
        try:
            graph_json = load_json(graph_path)
            graph_idx = graph_json.get("graph_index")
            if graph_idx is not None:
                graphs_cache[graph_idx] = graph_json
        except Exception as e:
            print(f"Error loading {graph_path}: {e}")
    
    print(f"Cached {len(graphs_cache)} graphs")
    
    # DEBUG: Test first case
    if len(inference_results) > 0:
        first_case = inference_results[0]
        graph_idx = first_case.get("graph_idx") or first_case.get("graph_index")
        if graph_idx in graphs_cache:
            debug_single_case(
                graphs_cache[graph_idx],
                first_case.get("evidence", {}),
                root_node_id,
                query_state
            )
    
    predictions = []
    ground_truths = []
    execution_times = []
    failed_cases = 0
    
    for i, case in enumerate(inference_results):
        if (i + 1) % 1000 == 0:
            print(f"Processing {i+1}/{len(inference_results)}")
        
        case_start = time.time()
        
        graph_idx = case.get("graph_idx") or case.get("graph_index")
        evidence_dict = case.get("evidence", {})
        
        # Extract ground truth
        ground_truth = case.get("predicted_root_prob")
        if ground_truth is None:
            prob_list = case.get("prob")
            if prob_list and isinstance(prob_list, list) and len(prob_list) > 0:
                ground_truth = prob_list[0]
        
        if ground_truth is None:
            failed_cases += 1
            continue
        
        if graph_idx not in graphs_cache:
            if verbose:
                print(f"  Case {i}: Graph {graph_idx} not found in cache")
            failed_cases += 1
            continue
        
        graph_json = graphs_cache[graph_idx]
        
        try:
            # Build model
            model, bn_states = build_bn_from_cpds(graph_json)
            
            # Perform inference
            predicted_prob = infer_root_prob(model, bn_states, root_node_id, 
                                            evidence_dict, query_state)
            
            case_end = time.time()
            exec_time = case_end - case_start
            
            predictions.append(predicted_prob)
            ground_truths.append(ground_truth)
            execution_times.append(exec_time)
            
            case["pomegranate_prediction"] = float(predicted_prob)
            case["execution_time"] = float(exec_time)
            
            if verbose and len(predictions) <= 5:
                print(f"  Case {i}: GT={ground_truth:.6f}, Pred={predicted_prob:.6f}, "
                      f"Time={exec_time:.4f}s")
        
        except Exception as e:
            if verbose:
                print(f"  Case {i} failed: {str(type(e).__name__)}: {str(e)[:100]}")
            failed_cases += 1
    
    # Calculate metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    if len(predictions) > 0:
        accuracy = np.mean(np.abs(predictions - ground_truths) < 0.05)
        mae = np.mean(np.abs(predictions - ground_truths))
        rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
        avg_time = np.mean(execution_times)
    else:
        accuracy = mae = rmse = avg_time = 0.0
    
    end_time = time.time()
    total_time = end_time - start_time
    
    metrics = {
        "config": {
            "num_graphs": num_graphs,
            "graphs_found": len(graphs_cache),
            "root_node_id": root_node_id,
            "query_state": query_state
        },
        "results": {
            "total_cases": len(inference_results),
            "processed_cases": len(predictions),
            "failed_cases": failed_cases,
            "accuracy_within_5_percent": float(accuracy),
            "mae": float(mae),
            "rmse": float(rmse),
            "avg_execution_time_per_case_seconds": float(avg_time),
            "total_execution_time_seconds": float(total_time)
        },
        "timing": {
            "start": start_time,
            "end": end_time,
            "duration_seconds": total_time
        }
    }
    
    save_json(inference_results, output_path)
    save_json(metrics, metrics_path)
    
    print("\n" + "=" * 70)
    print("POMEGRANATE INFERENCE SUMMARY")
    print("=" * 70)
    print(f"Graphs: {len(graphs_cache)} found (config expects {num_graphs})")
    print(f"Cases: {metrics['results']['total_cases']} total")
    print(f"Processed: {metrics['results']['processed_cases']} | Failed: {metrics['results']['failed_cases']}")
    print(f"\nMetrics:")
    print(f"  Accuracy (within 5%): {metrics['results']['accuracy_within_5_percent']:.4f}")
    print(f"  MAE: {metrics['results']['mae']:.6f}")
    print(f"  RMSE: {metrics['results']['rmse']:.6f}")
    print(f"\nTiming:")
    print(f"  Avg time/case: {metrics['results']['avg_execution_time_per_case_seconds']:.6f}s")
    print(f"  Total time: {metrics['results']['total_execution_time_seconds']:.2f}s")
    print("=" * 70)
    print(f"Predictions: {output_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()