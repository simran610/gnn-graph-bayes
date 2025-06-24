# File: bayesian_network_builder.py
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
            cpd = TabularCPD(variable=node, variable_card=card, values=[[0.5], [0.5]])
        else:
            parent_card = [card] * len(parents)
            alpha = np.ones(card)  # Dirichlet parameters
            num_cols = card ** len(parents)
            values = np.array([np.random.dirichlet(alpha) for _ in range(num_cols)]).T
            #values = np.random.rand(card, card**len(parents))
            #values /= values.sum(axis=0)  # Normalize
            cpd = TabularCPD(variable=node, variable_card=card,
                             values=values.tolist(), evidence=parents, evidence_card=parent_card)
        model.add_cpds(cpd)

    assert model.check_model()
    return model
