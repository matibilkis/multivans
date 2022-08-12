import numpy as np
from ast import literal_eval
import pandas as pd


def kill_and_simplify(cdb, initial_cost, killer, simplifier, max_rounds = 1000):
    killed_db, cost, murders = killer.remove_irrelevant_gates(initial_cost,cdb)
    simplified_db, ns =  simplifier.reduce_circuit(killed_db)
    ops = ns+murders
    for it in range(max_rounds):
        killed_db, cost, murders = killer.remove_irrelevant_gates(cost,simplified_db)
        simplified_db, ns =  simplifier.reduce_circuit(killed_db)
        ops += ns+murders
        if (murders == 0):
            break ###this means you enter a loop in the simplifier...
    return simplified_db, cost,ops



def reindex_symbol(list_of_symbols, first_symbol_number):
    reindexed=[]
    for ind, sym in enumerate(list_of_symbols):
        if sym == [] or sym is None:
            reindexed.append(None)
        else:
            reindexed.append("th_{}".format(int(sym.replace("th_",""))+first_symbol_number))
    return reindexed


def get_qubits_involved(translator, circuit_db):
    inds = list(circuit_db["ind"].values)
    counts = {q:0 for q in range(translator.n_qubits)}
    for k in inds:
        if k < translator.number_of_cnots:
            c, t = translator.indexed_cnots[str(k)]
            counts[c]+=1
            counts[t]+=1
        else:
            cont = k-translator.number_of_cnots
            qq = (k-translator.number_of_cnots)%translator.n_qubits
            counts[qq]+=1

    active_qubits = []
    for q,k in enumerate(list(counts.values())):
        if k>0:
            active_qubits.append(q)
    return active_qubits

def shift_symbols_up(idinserter, indice, circuit_db):
    """
    indice is the place at which the gate was added.
    """
    for k in range(indice+2, circuit_db.shape[0]):
        if circuit_db.loc[k]["ind"] < idinserter.number_of_cnots or type(circuit_db.loc[k]["symbol"]) != str:
            pass
        else:
            old_value = circuit_db.loc[k]["symbol"]
            number_symbol = int(old_value.replace("th_","")) +1
            new_value = "th_{}".format(number_symbol)
            circuit_db.loc[k] = circuit_db.loc[k].replace(to_replace=old_value,value=new_value)
    return circuit_db

def shift_symbols_down(simplifier, indice, circuit_db):
    """
    indice is the place at which the gate was added.
    """
    for k in range(indice, circuit_db.shape[0]):
        if circuit_db.loc[k]["ind"] < simplifier.number_of_cnots or type(circuit_db.loc[k]["symbol"]) != str:
            pass
        else:
            old_value = circuit_db.loc[k]["symbol"]
            number_symbol = int(old_value.replace("th_","")) -1
            new_value = "th_{}".format(number_symbol)
            circuit_db.loc[k] = circuit_db.loc[k].replace(to_replace=old_value,value=new_value)
    return circuit_db




def check_symbols_ordered(circuit_db):
    symbol_int = list(circuit_db["symbol"].dropna().apply(lambda x: int(x.replace("th_",""))))
    return symbol_int == sorted(symbol_int)

def order_symbol_labels(circuit_db):
    """
    it happens that when a circuit is simplified, symbol labels get unsorted. This method corrects that (respecting the ordering in the gates)
    """
    if check_symbols_ordered(circuit_db) is False:
        inns = circuit_db["symbol"].dropna().index
        filtered_db = circuit_db.loc[inns]["symbol"].astype(str)
        news = ["th_{}".format(k) for k in np.sort(list(circuit_db["symbol"].dropna().apply(lambda x: int(x.replace("th_","")))))]
        sss = pd.Series(news, index=inns)
        nans = circuit_db["symbol"][circuit_db["symbol"].isna()]
        ser = pd.concat([nans,sss])
        ser = ser.sort_index()
        circuit_db = circuit_db.drop(["symbol"], axis=1)
        circuit_db.insert(loc=1, column="symbol",value=ser)
    return circuit_db

def type_get(x, translator):
    return (x-translator.number_of_cnots)//translator.n_qubits




def check_rot(ind_gate, translator):
    return translator.number_of_cnots<= ind_gate <(3*translator.n_qubits + translator.number_of_cnots)

def check_cnot(ind_gate, translator):
    return translator.number_of_cnots> ind_gate# <(3*translator.n_qubits + translator.number_of_cnots)



### killer

def qubit_get(x, translator):
    return (x-translator.number_of_cnots)%translator.n_qubits



def u2zxz(U, with_phase=False):
    """
    U = e^i \delta RZ(\alpha) RX(\theta) Rz(\beta)
    returns U (decomposed as such, to check) and [\delta, \alpha, \theta, \beta].
    note we just change of basis and apply zyz decomposition.

    We won't be able to write, in general,
    """
    th = 2*np.arccos(np.abs(U[0,0]))
    beta = np.angle(U[0,0]) - np.angle(U[0,1]) - np.pi/2
    delta = .5*(np.angle(U[0,0]) + np.angle(U[1,1]))
    alpha = np.angle(U[0,1]) - np.angle(U[1,1]) + np.pi/2

    if with_phase==False:
        return [alpha,th, beta]
    else:
        rz_alpha = np.diag([np.exp(1j*alpha/2), np.exp(-1j*alpha/2)])
        rz_beta = np.diag([np.exp(1j*beta/2), np.exp(-1j*beta/2)])
        rx_th = np.array([[np.cos(th/2), -1j*np.sin(th/2)],[-1j*np.sin(th/2), np.cos(th/2)]])
        r = np.exp(1j*delta)*rz_alpha.dot(rx_th).dot(rz_beta)

        return r,[delta, alpha, th, beta]
