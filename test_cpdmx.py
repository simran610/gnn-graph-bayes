    # for i, (tree, model) in enumerate(graph_model_pairs):
    #     for cpd in model.get_cpds():
    #         flat = flatten(cpd.values.tolist())
    #         if len(flat) == global_max_len:
    #             print(f" Graph #{i}, Node: {cpd.variable}")
    #             print(f"Flattened CPD length: {len(flat)}")
    #             print(" Flattened CPD:")
    #             print(flat)
    #             print(f"\n Padded CPD (length = {global_max_len}):")
    #             print(pad_cpd_values(flat, global_max_len))

    #             # Also print table version
    #             variable = cpd.variable
    #             var_card = cpd.variable_card
    #             evidence = cpd.get_evidence()
    #             evidence_card = cpd.cardinality[1:]

    #             if evidence:
    #                 parent_combos = list(itertools.product(*[range(card) for card in evidence_card]))
    #                 rows = []
    #                 cpd_reshaped = cpd.values.reshape(var_card, -1)
    #                 for j, combo in enumerate(parent_combos):
    #                     for state in range(var_card):
    #                         row = list(combo) + [state, cpd_reshaped[state, j]]
    #                         rows.append(row)
    #                 columns = evidence + [f"{variable}_state", "P"]
    #                 df = pd.DataFrame(rows, columns=columns)
    #             else:
    #                 df = pd.DataFrame({
    #                     f"{variable}_state": list(range(var_card)),
    #                     "P": cpd.values.flatten()
    #                 })

    #             print(f"\n Tabular CPD for {cpd.variable}:")
    #             print(df)

    #             # Exit after first match (optional, remove if you want all max-length CPDs)
    #             break
    #     else:
    #         continue
    #     break  # exits outer loop too