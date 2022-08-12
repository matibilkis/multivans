dev = qml.device("default.qubit", wires=translator.n_qubits)
@qml.qnode(dev)
def qnode(inputs, weights,**kwargs):
    """
    weights is a list of variables (automatic in penny-lane, here i feed [] so i don't update parameter values)
    """
    symbols = database.get_trainable_symbols(translator,cinputs)
    ww = {s:w for s,w in zip( symbols, weights)}
    cinputs = database.update_circuit_db_param_values(translator, cinputs, ww)

    list_of_gate_ids = [templates.gate_template(**dict(cinputs.iloc[k])) for k in range(len(cinputs))]
    for i,gate_id in enumerate(list_of_gate_ids):
        translator.db = translator.append_to_circuit(translator.db, gate_id)
    return [qml.expval(qml.PauliZ(k)) for k in range(translator.n_qubits)]
