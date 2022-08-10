
import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import reload
import tensorflow as tf
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from tqdm import tqdm
import utilities.translator.pennylane_translator as penny_translator
import utilities.evaluator.pennylane_evaluator as penny_evaluator
import utilities.variational.pennylane_model as penny_variational
import utilities.simplification.simplifier as penny_simplifier
import utilities.simplification.misc as simplification_misc#.kill_and_simplify
import utilities.simplification.gate_killer as penny_killer
import utilities.database.database as database
import utilities.database.templates as templates
import utilities.mutator.idinserter as idinserter
import running.misc as miscrun
from utilities.simplification.misc import get_qubits_involved, reindex_symbol, shift_symbols_down, shift_symbols_up, type_get, check_rot, order_symbol_labels, check_cnot, check_symbols_ordered
import coding.coding_simplifier as simplli






reload(penny_simplifier)
reload(templates)
reload(penny_translator)
reload(miscrun)
reload(idinserter)
reload(penny_variational)

n_qubits = 4

translator = penny_translator.PennyLaneTranslator(n_qubits = n_qubits, initialize="x")
simplifier = simplli.PennyLane_Simplifier(translator)
db = database.concatenate_dbs([templates.x_layer(translator, params=True),templates.z_layer(translator, params=True) ]*2)
c, cdb = translator.give_circuit(db)
simplified_db = cdb.copy()





gates_on_qubit, on_qubit_order = simplifier.get_positional_dbs(_, cdb)
simplification = False
for q, qubit_gates_path in gates_on_qubit.items():
    if simplification is True:
        break
    for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-2]):

        if simplification is True:
            break
        ind_gate_p1 = qubit_gates_path[order_gate_on_qubit+1]
        ind_gate_p2 = qubit_gates_path[order_gate_on_qubit+2]

        if (check_rot(ind_gate, simplifier.translator) == True) and (check_rot(ind_gate_p1, simplifier.translator) == True) and (check_rot(ind_gate_p2, simplifier.translator) == True):

            type_0 = type_get(ind_gate,simplifier.translator)
            type_1 = type_get(ind_gate_p1,simplifier.translator)
            type_2 = type_get(ind_gate_p2,simplifier.translator)

            if type_0 == type_2:
                types = [type_0, type_1, type_2]
                for next_order_gate_on_qubit, ind_gate_next in enumerate(qubit_gates_path[order_gate_on_qubit+3:]):
                    if (check_rot(ind_gate_next, simplifier.translator) == True):# and (next_order_gate_on_qubit < len(qubit_gates_path[order_gate_on_qubit+3:])):
                        types.append(type_get(ind_gate_next, simplifier.translator))
                        simplification=True
                    else:
                        break
                if simplification == True:
                    break


indices_to_compile = [on_qubit_order[q][order_gate_on_qubit+k] for k in range(len(types))]
simplifier.translator_ =  penny_translator.PennyLaneTranslator(n_qubits = 2)

u_to_compile_db = simplified_db.loc[indices_to_compile]
u_to_compile_db["ind"] = simplifier.translator_.n_qubits*type_get(u_to_compile_db["ind"], simplifier.translator) + simplifier.translator_.number_of_cnots
u_to_compile_db["symbol"] = None ##just to be sure it makes no interference with the compiler...
cir, cirdb = simplifier.translator_.give_circuit(u_to_compile_db)

compile_circuit_db = penny_simplifier.construct_compiling_circuit(simplifier.translator_, u_to_compile_db)


c, cdb = simplifier.translator_.give_circuit(compile_circuit_db)

minimizer = penny_variational.PennyModel(simplifier.translator_,lr=0.1, patience=50, hamiltonian="Z")
minimized_db, [cost, resolver, history_training] = minimizer.variational(epochs=100, verbose = 1)


resolver

OneQbit_translator = penny_translator.PennyLaneTranslator(n_qubits=1)
u1s = templates.u1_db(OneQbit_translator, 0, params=True)
u1s["param_value"] = -np.array(list(resolver.values()))
resu_comp, resu_db = OneQbit_translator.give_circuit(u1s)


tdb = simplified_db.loc[indices_to_compile]
tdb["ind"] = simplifier.translator_.n_qubits*type_get(tdb["ind"], simplifier.translator) + OneQbit_translator.number_of_cnots
tdb["symbol"] = None

target_c, target_db = OneQbit_translator.give_circuit(tdb)



resu_comp.device.state
target_c.device.state



u_to_compile_db_1q = u_to_compile_db.copy()
u_to_compile_db_1q["ind"] = u_to_compile_db["ind"] = type_get(u_to_compile_db["ind"], self.translator_)




plt.plot(history_training.history["cost"])







minimizer = Minimizer(self.translator_, mode="compiling", hamiltonian="Z")

cost, resolver, history = minimizer.minimize([compile_circuit], symbols=self.translator.get_trainable_symbols(compile_circuit_db))

OneQbit_translator = CirqTranslater(n_qubits=1)
u1s = u1_db(OneQbit_translator, 0, params=True)
u1s["param_value"] = -np.array(list(resolver.values()))
resu_comp, resu_db = OneQbit_translator.give_circuit(u1s,unresolved=False)


u_to_compile_db_1q = u_to_compile_db.copy()
u_to_compile_db_1q["ind"] = u_to_compile_db["ind"] = type_get(u_to_compile_db["ind"], self.translator_)

cc, cdb = OneQbit_translator.give_circuit(u_to_compile_db_1q, unresolved=False)
c = cc.unitary()
r = resu_comp.unitary()

## phase_shift if necessary
if np.abs(np.mean(c/r) -1) > 1:
    u1s.loc[0] = u1s.loc[0].replace(to_replace=u1s["param_value"][0], value=u1s["param_value"][0] + 2*np.pi)# Rz(\th) = e^{-ii \theta \sigma_z / 2}c0, cdb0 = self.translator.give_circuit(pd.DataFrame([gate_template(0, param_value=2*np.pi)]), unresolved=False)
resu_comp, resu_db = self.translator.give_circuit(u1s,unresolved=False)

first_symbols = simplified_db["symbol"][indices_to_compile][:3]

for new_ind, typ, pval in zip(indices_to_compile[:3],[0,1,0], list(u1s["param_value"])):
    simplified_db.loc[new_ind+0.1] = gate_template(self.translator.number_of_cnots + q + typ*self.translator.n_qubits,
                                                     param_value=pval, block_id=simplified_db.loc[new_ind]["block_id"],
                                                     trainable=True, symbol=first_symbols[new_ind])

for old_inds in indices_to_compile:
    simplified_db = simplified_db.drop(labels=[old_inds],axis=0)#

simplified_db = simplified_db.sort_index().reset_index(drop=True)
killed_indices = indices_to_compile[3:]
db_follows = original_db[original_db.index>indices_to_compile[-1]]


simplified_db = self.order_symbols(simplified_db)
# if len(db_follows)>0:
#     gates_to_lower = list(db_follows.index)
#     number_of_shifts = len(killed_indices)
#     for k in range(number_of_shifts):
#         simplified_db = shift_symbols_down(self.translator, gates_to_lower[0]-number_of_shifts, simplified_db)


























indices_to_compile = [on_qubit_order[q][order_gate_on_qubit+k] for k in range(len(types))]
self.translator_ = PennyLaner(n_qubits=2)
u_to_compile_db = simplified_db.loc[indices_to_compile]
u_to_compile_db["ind"] = self.translator_.n_qubits*type_get(u_to_compile_db["ind"], self.translator) + self.translator_.number_of_cnots
u_to_compile_db["symbol"] = None ##just to be sure it makes no interference with the compiler...

compile_circuit, compile_circuit_db = construct_compiling_circuit(self.translator_, u_to_compile_db)
minimizer = Minimizer(self.translator_, mode="compiling", hamiltonian="Z")

cost, resolver, history = minimizer.minimize([compile_circuit], symbols=self.translator.get_trainable_symbols(compile_circuit_db))

OneQbit_translator = CirqTranslater(n_qubits=1)
u1s = u1_db(OneQbit_translator, 0, params=True)
u1s["param_value"] = -np.array(list(resolver.values()))
resu_comp, resu_db = OneQbit_translator.give_circuit(u1s,unresolved=False)


u_to_compile_db_1q = u_to_compile_db.copy()
u_to_compile_db_1q["ind"] = u_to_compile_db["ind"] = type_get(u_to_compile_db["ind"], self.translator_)

cc, cdb = OneQbit_translator.give_circuit(u_to_compile_db_1q, unresolved=False)
c = cc.unitary()
r = resu_comp.unitary()

## phase_shift if necessary
if np.abs(np.mean(c/r) -1) > 1:
    u1s.loc[0] = u1s.loc[0].replace(to_replace=u1s["param_value"][0], value=u1s["param_value"][0] + 2*np.pi)# Rz(\th) = e^{-ii \theta \sigma_z / 2}c0, cdb0 = self.translator.give_circuit(pd.DataFrame([gate_template(0, param_value=2*np.pi)]), unresolved=False)
resu_comp, resu_db = self.translator.give_circuit(u1s,unresolved=False)

first_symbols = simplified_db["symbol"][indices_to_compile][:3]

for new_ind, typ, pval in zip(indices_to_compile[:3],[0,1,0], list(u1s["param_value"])):
    simplified_db.loc[new_ind+0.1] = gate_template(self.translator.number_of_cnots + q + typ*self.translator.n_qubits,
                                                     param_value=pval, block_id=simplified_db.loc[new_ind]["block_id"],
                                                     trainable=True, symbol=first_symbols[new_ind])

for old_inds in indices_to_compile:
    simplified_db = simplified_db.drop(labels=[old_inds],axis=0)#

simplified_db = simplified_db.sort_index().reset_index(drop=True)
killed_indices = indices_to_compile[3:]
db_follows = original_db[original_db.index>indices_to_compile[-1]]


simplified_db = self.order_symbols(simplified_db)
