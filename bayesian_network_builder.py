# bayesian_network_builder.py
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np

def build_bn_from_tree(G, config):
    model = DiscreteBayesianNetwork()
    model.add_edges_from(G.edges())
    
    card = 2  # Default to binary variables
    for node in G.nodes():
        parents = list(G.predecessors(node))
        if not parents:
            if np.random.random() < 0.5:
                root_prob = np.random.beta(2, 5)  # Low prob
            else:
                root_prob = np.random.beta(5, 2)  # High prob

            cpd = TabularCPD(variable=node, variable_card=card, 
                           values=[[1-root_prob], [root_prob]])
        else:
            parent_card = [card] * len(parents)
            num_cols = card ** len(parents)
            
            values = np.zeros((card, num_cols))
            
            for col in range(num_cols):
                # Convert column to parent configuration
                parent_config = [(col >> i) & 1 for i in range(len(parents))]
                
                # OR-like relationship with noise
                # base_prob = 0.8 if any(parent_config) else 0.2
                # # Add controlled noise
                # prob_true = np.clip(base_prob + np.random.normal(0, 0.1), 0.1, 0.9)
                
                parent_sum = sum(p for p in parent_config)
                #base_prob = 0.6 + 0.2 * parent_sum / len(parent_config)
                base_prob = 0.3 + 0.4 * parent_sum / len(parent_config) 
                noise = np.random.normal(0, 0.2)
                prob_true = np.clip(base_prob + noise, 0.1, 0.9)


                values[0, col] = 1 - prob_true  # P(node=0|parents)
                values[1, col] = prob_true      # P(node=1|parents)
            
            cpd = TabularCPD(variable=node, variable_card=card,
                           values=values.tolist(), evidence=parents, 
                           evidence_card=parent_card)
        model.add_cpds(cpd)
    
    assert model.check_model()
    return model

