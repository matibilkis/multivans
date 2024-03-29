import numpy as np

import tensorflow as tf
import utilities.translator.pennylane_translator as penny_translator
import utilities.database.database as database
import utilities.database.templates as templates
from utilities.database.database import concatenate_dbs
from utilities.simplification.misc import shift_symbols_down, qubit_get, get_qubits_involved, order_symbol_labels
# import utilities.variational.pennylane_model as penny_variational
import utilities.variational.tfq.variational as minimizer

class GateKiller:
    def __init__(self,
                translator,
                translator_test,
                **kwargs):
        """
        """
        self.translator = translator
        self.test_translator = translator_test
        self.test_model = minimizer.Minimizer(translator,mode="VQE",hamiltonian=kwargs.get("hamiltonian","XXZ"),params=kwargs.get("params",[1.,.01]), lower_bound_cost=0., who="killer")

        #self.max_relative_increment = kwargs.get("max_relative_increment", 0.05)
        self.accept_wall = kwargs.get("accept_wall", 1e5)
        self.printing = True

    def get_positional_dbs(self, circuit_db):
        """
        this is here to check whether to leave block without gates or not
        """
        circuit, circuit_db = self.test_translator.give_circuit(circuit_db)
        qubits_involved = get_qubits_involved(self.test_translator, circuit_db)

        gates_on_qubit = {q:[] for q in qubits_involved}
        on_qubit_order = {q:[] for q in qubits_involved}

        for order_gate, ind_gate in enumerate( circuit_db["ind"]):
            if ind_gate < self.translator.number_of_cnots:
                control, target = self.translator.indexed_cnots[str(ind_gate)]
                gates_on_qubit[control].append(ind_gate)
                gates_on_qubit[target].append(ind_gate)
                on_qubit_order[control].append(order_gate)
                on_qubit_order[target].append(order_gate)
            else:
                gates_on_qubit[(ind_gate-self.translator.n_qubits)%self.translator.n_qubits].append(ind_gate)
                on_qubit_order[(ind_gate-self.translator.n_qubits)%self.translator.n_qubits].append(order_gate)
        return gates_on_qubit, on_qubit_order


    def remove_irrelevant_gates(self,initial_cost, circuit_db):
        first_cost = initial_cost
        number_of_gates = len(database.get_trainable_params_value(self.test_translator,circuit_db))
        for murder_attempt in range(number_of_gates):
            circuit_db, new_cost, killed = self.kill_one_unitary(first_cost, circuit_db)
            circuit_db = order_symbol_labels(circuit_db)
            # print("Killing, try {}/{}. Killed was {} now cost changes by: {}%".format(murder_attempt, number_of_gates, killed,(initial_cost-new_cost)/np.abs(initial_cost)))
            if killed == False:
                break
            if (self.printing == True) and (killed == True):
                print("kill 1qbit gate, try {}/{}. Increased by: {}%".format(murder_attempt, number_of_gates, (initial_cost-new_cost)/np.abs(initial_cost)))
        return circuit_db, new_cost, murder_attempt

    def kill_one_unitary(self, initial_cost, circuit_db):

        blocks = list(set(circuit_db["block_id"]))
        for b in self.translator.untouchable_blocks:
            blocks.remove(b)

        candidates = []
        for b in blocks:
            block_db = circuit_db[circuit_db["block_id"] == b]
            block_db_trainable = block_db[block_db["trainable"] == True]
            block_db_trainable = block_db_trainable[~block_db_trainable["symbol"].isna()]
            block_db_trainable = block_db_trainable[~block_db_trainable["symbol"].isna()]
            all_candidates = list(block_db_trainable.index)

            ### check if the circuit is too short... (another possibility is to replace this guy by an rz(0)
            gates_on_qubit, on_qubit_order = self.get_positional_dbs(block_db_trainable)
            for kg in all_candidates:
                qubit_affected = qubit_get(block_db_trainable.loc[kg]["ind"], self.translator)
                if len(gates_on_qubit[qubit_affected]) >= 2:
                    candidates.append(kg)

        killed_costs = []

        for index_candidate in candidates:
            killed_circuit_db = circuit_db.copy()
            killed_circuit_db = killed_circuit_db.drop(labels=[index_candidate])
            killed_circuit_db = shift_symbols_down(self.test_translator, index_candidate+1, killed_circuit_db)

            killed_costs.append(self.test_model.build_and_give_cost(killed_circuit_db))

        relative_increments = (np.array(killed_costs)-initial_cost)/np.abs(initial_cost)
        # print(relative_increments)
        if len(relative_increments) == 0:
            return circuit_db, initial_cost, False

        #if np.min(relative_increments) < self.max_relative_increment:
        if self.accepting_criteria(initial_cost, np.min(killed_costs)) == True:
            #pos_min = np.argmin(relative_increments)
            pos_min = np.argmin(killed_costs)
            index_to_kill = candidates[pos_min]
            new_cost = killed_costs[pos_min]

            killed_circuit_db = circuit_db.copy()
            killed_circuit_db = killed_circuit_db.drop(labels=[index_to_kill])
            killed_circuit_db = killed_circuit_db.sort_index().reset_index(drop=True)
            killed_circuit_db = shift_symbols_down(self.test_translator, index_to_kill, killed_circuit_db)
            return killed_circuit_db, new_cost, True
        else:
            return circuit_db, initial_cost, False


    def accepting_criteria(self, reference_cost, new_cost):
        """
        if decreases energy, we accept it;
        otherwise exponentially decreasing probability of acceptance (the 100 is yet another a bit handcrafted)
        """
        #return  < 0.01
        relative_error = (new_cost-reference_cost)/np.abs(reference_cost)
        if new_cost <= reference_cost:
            return True
        else:
            return np.random.random() < np.exp(-np.abs(relative_error)*self.accept_wall)
