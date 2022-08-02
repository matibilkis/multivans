import numpy as np

def gate_counter_on_qubits(translator, circuit_db, untouchable_qubits = []):
    """
    Gives gate count for each qbit. First entry rotations, second CNOTS
    """

    touchable_qubits = list(range(translator.n_qubits))
    for q in untouchable_qubits:
        touchable_qubits.remove(q)
    ngates = {k:[0,0] for k in touchable_qubits}
    for ind in circuit_db["ind"]:
        if ind < translator.number_of_cnots:
            control, target = translator.indexed_cnots[str(ind)]
            if (control in touchable_qubits) and (target in touchable_qubits):
                ngates[control][1]+=1
                ngates[target][1]+=1
        else:
            qind = (ind-translator.number_of_cnots)%translator.n_qubits
            if qind in touchable_qubits:
                ngates[qind][0]+=1
    return np.array(list(ngates.values()))


def get_symbol_number_from(insertion_index, circuit_db):
    ### check symbol number ###
    symbol_found=False
    for k in range(0, insertion_index+1)[::-1]:
        if type(circuit_db.loc[k]["symbol"]) == str:
            number_symbol = int(circuit_db.loc[k]["symbol"].replace("th_","")) +1
            symbol_found=True
            break
    if not symbol_found:
        number_symbol = 0
    return number_symbol
