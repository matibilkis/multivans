import tensorflow as tf
from pennylane import numpy as np
import pennylane as qml
import pandas as pd
#import utilities.templates as templates
import utilities.database.database as database
import utilities.database.templates as templates


class PennyLaneTranslator:
    def __init__(self, n_qubits, **kwargs):
        self.n_qubits = n_qubits
        #self.qubits = cirq.GridQubit.rect(1, n_qubits)

        ### blocks that are fixed-structure (i.e. channels, state_preparation, etc.)
        untouchable_blocks = kwargs.get("untouchable_blocks",[])
        self.untouchable_blocks = untouchable_blocks
        self.discard_qubits = kwargs.get("discard_qubits",[]) ###these are the qubits that you don't measure, i.e. environment


        #### keep a register on which integers corresponds to which CNOTS, target or control.
        self.indexed_cnots = {}
        self.cnots_index = {}
        count = 0
        for control in range(self.n_qubits):
            for target in range(self.n_qubits):
                if control != target:
                    self.indexed_cnots[str(count)] = [control, target]
                    self.cnots_index[str([control,target])] = count
                    count += 1
        self.number_of_cnots = len(self.indexed_cnots)
        #
        self.cgates = {0:qml.RZ, 1: qml.RX, 2:qml.RY}
        self.temp_circuit_db = {}

    def spit_gate(self,gate_id):
        ## the symbols are all elements we added but the very last one (added on the very previous line)
        ind = gate_id["ind"]

        if ind<self.number_of_cnots:
            qml.CNOT(self.indexed_cnots[str(ind)])
        else:
            cgate_type = (ind-self.number_of_cnots)%3
            qubit = (ind-self.number_of_cnots)//self.n_qubits
            symbol_name = gate_id["symbol"]
            param_value = gate_id["param_value"]
            self.cgates[cgate_type](param_value,qubit)

    def append_to_circuit(self,circuit_db, gate_id):
        ## the symbols are all elements we added but the very last one (added on the very previous line)
        symbols = []
        for j in [k["symbol"] for k in circuit_db.values()]:
            if j != None:
                symbols.append(j)
        ind = gate_id["ind"]

        gate_index = len(list(circuit_db.keys()))
        circuit_db[gate_index] = gate_id #this is the new item to add

        if ind<self.number_of_cnots:
            qml.CNOT(self.indexed_cnots[str(ind)])
        else:
            cgate_type = (ind-self.number_of_cnots)%3
            qubit = (ind-self.number_of_cnots)//self.n_qubits
            symbol_name = gate_id["symbol"]
            param_value = gate_id["param_value"]
            if symbol_name is None:
                symbol_name = "th_"+str(len(symbols))
                circuit_db[gate_index]["symbol"] = symbol_name
            else:
                if symbol_name in symbols:
                    print("Symbol repeated {}, {}".format(symbol_name, symbols))
            self.cgates[cgate_type](param_value,qubit)
        return circuit_db

    def give_circuit(self, dd,**kwargs):
        """

        """
        unresolved = kwargs.get("unresolved",True)

        dev = qml.device("default.qubit", wires=self.n_qubits)
        ### TO-DO: CHECK INPUT COPY!
        @qml.qnode(dev)
        def qnode(inputs, weights,**kwargs):
            """
            weights is a list of variables (automatic in penny-lane, here i feed [] so i don't update parameter values)
            """

#            db = kwargs.get("db",{})
            self.db = {}
            cinputs = inputs#.copy()
            symbols = database.get_trainable_symbols(self,cinputs)
            ww = {s:w for s,w in zip( symbols, weights)}
            cinputs = database.update_circuit_db_param_values(self, cinputs, ww)

            list_of_gate_ids = [templates.gate_template(**dict(cinputs.iloc[k])) for k in range(len(cinputs))]
            for i,gate_id in enumerate(list_of_gate_ids):
                self.db = self.append_to_circuit(self.db, gate_id)
            return [qml.expval(qml.PauliZ(k)) for k in range(self.n_qubits)]

        #self.db = {}
        circuit = qnode(dd, []) ##this creates the database; the weights are used as trainable variables # of optimization
        circuit_db = pd.DataFrame.from_dict(self.db,orient="index")
        self.db = circuit_db.copy()
        self.db_train = self.db.copy() ### copy to be used in PennyLaneModel
        return qnode, circuit_db
