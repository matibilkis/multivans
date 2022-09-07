import os
import sys
sys.path.insert(0, os.getcwd())

import tensorflow as tf
# from pennylane import numpy as np
# import pennylane as qml
import pandas as pd
import utilities.database.database as database
import utilities.database.templates as templates

import cirq
import pandas as pd
import numpy as np
import cirq
import sympy

class TFQTranslator:
    def __init__(self, n_qubits, **kwargs):
        """
        class that translates database to cirq circuits
        """
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)

        ### blocks that are fixed-structure (i.e. channels, state_preparation, etc.)
        untouchable_blocks = kwargs.get("untouchable_blocks",[])
        self.untouchable_blocks = untouchable_blocks
        self.discard_qubits = kwargs.get("discard_qubits",[]) ###these are the qubits that you don't measure, i.e. environment

        self.noisy = kwargs.get("noisy",False)
        if self.noisy==True:
            self.noise_model = kwargs.get("noise_model","noisy") ###
            if self.noise_model.lower() == "depolarizing": ###just because i have data already in this regard
                self.noise_model = "noisy"
        self.noise_strength = kwargs.get("noise_strength",0.01)

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
        self.cgates = {0:cirq.rz, 1: cirq.rx, 2:cirq.ry}


    def append_to_circuit(self, gate_id, circuit, circuit_db, **kwargs):
        """
        adds gate_id instructions to current circuit. Returns new circuit (cirq object) and new circuit_db (pd.DataFrame)

        gate_id: dictionary containing gate info to append
        circuit: Cirq.Circuit object
        circuit_db: pandas DataFrame (all circuit info is appended to here)
        """
        unresolved = kwargs.get("unresolved",False)
        ind = gate_id["ind"]
        gate_index = len(list(circuit_db.keys()))
        circuit_db[gate_index] = gate_id #this is the new item to add

        ## the symbols are all elements we added but the very last one (added on the very previous line)
        symbols = []
        for j in [k["symbol"] for k in circuit_db.values()][:-1]:
            if j != None:
                symbols.append(j)
        ## custom gate
        if ind == -1: ## warning, no support on TFQ for the moment...
            circuit_db[gate_index]["symbol"] = None
            u=gate_id["param_value"]   ##param_value will be the unitary (np.array)
            q=gate_id["qubits"] #list
            qubits = circuit_db[gate_index]["qubits"]
            uu = cirq.MatrixGate(u)
            circuit.append(uu.on(*[self.qubits[qq] for qq in qubits]))
            return circuit, circuit_db
        #### add CNOT
        elif 0 <= ind < self.number_of_cnots:
            control, target = self.indexed_cnots[str(ind)]
            circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            circuit_db[gate_index]["symbol"] = None
            return circuit, circuit_db

        ### add HADDAMARD
        elif self.number_of_cnots + 3*self.n_qubits <= ind < self.number_of_cnots + 4*self.n_qubits:
            qubit  = (ind-self.number_of_cnots)%self.n_qubits
            circuit.append(cirq.H.on(self.qubits[qubit]))
            circuit_db[gate_index]["symbol"] = None
            return circuit, circuit_db

        #### add rotation
        elif self.number_of_cnots <= ind < self.number_of_cnots + 3*self.n_qubits:
            gate_type_index = (ind - self.number_of_cnots)//self.n_qubits
            qubit  = (ind-self.number_of_cnots)%self.n_qubits
            gate = self.cgates[gate_type_index]

            symbol_name = gate_id["symbol"]
            param_value = gate_id["param_value"]

            if symbol_name is None:
                symbol_name = "th_"+str(len(symbols))
                circuit_db[gate_index]["symbol"] = symbol_name
            else:
                if symbol_name in symbols:
                    print("warning, repeated symbol while constructing the circuit, see circuut_\n  symbol_name {}\n symbols {}\ncircuit_db {} \n\n\n".format(symbol_name, symbols, circuit_db))
            if (param_value is None) or (unresolved is True):
                if gate_id["trainable"] == True: ##only leave unresolved those gates that will be trianed
                    param_value = sympy.Symbol(symbol_name)
            circuit.append(gate(param_value).on(self.qubits[qubit]))
            return circuit, circuit_db

        else:
            raise AttributeError("Wrong index!", ind)

    def give_circuit(self, dd,**kwargs):
        """
        retrieves circuit from circuit_db. It is assumed that the order in which the symbols are labeled corresponds to order in which their gates are applied in the circuit.
        If unresolved is False, the circuit is retrieved with the values of rotations (not by default, since we feed this to a TFQ model)
        """
        unresolved = kwargs.get("unresolved",True)
        just_call = kwargs.get("just_call",False)

        list_of_gate_ids = [templates.gate_template(**dict(dd.iloc[k])) for k in range(len(dd))]
        circuit, circuit_db = [],{}
        for k in list_of_gate_ids:
            circuit , circuit_db = self.append_to_circuit(k,circuit, circuit_db, unresolved=unresolved)
        circuit = cirq.Circuit(circuit)
        circuit_db = pd.DataFrame.from_dict(circuit_db,orient="index")
        #### we make sure that the symbols appearing correspond to the ordering in which we add the gate to the circuit
        if just_call == False:
            self.db = circuit_db.copy()
            self.db_train = self.db.copy() ### copy to be used in PennyLaneModel

        if self.noisy == True:
            if self.noise_model == "noisy":

                noisy_circuit = []
                for k in list(circuit.all_operations()):
                    for q in k.qubits:
                        noisy_circuit.append(cirq.depolarize(self.noise_strength).on(q))
                    noisy_circuit.append(k)
            else:
                noisy_circuit = []

                ### State preparation error
                for k in circuit.all_qubits():
                    noisy_circuit.append(cirq.BitFlipChannel(p=self.noise_strength*1e-2).on(k))

                ## Fig 2.b ArXiv 2101.02109 (AmplitudeDamping instead of reset channel)
                for k in list(circuit.all_operations()):
                    if len(k.qubits) == 1:
                        depo_strength = self.noise_strength*1e-5
                        for q in k.qubits:
                            noisy_circuit.append(cirq.DepolarizingChannel(depo_strength*self.noise_strength).on(q))
                            noisy_circuit.append(cirq.PhaseFlipChannel(self.noise_strength*1e-3).on(q))
                            noisy_circuit.append(cirq.AmplitudeDampingChannel(self.noise_strength*1e-3).on(q))
                    elif len(k.qubits) == 2:
                        depo_strength = self.noise_strength*1e-5
                        for ind,q in enumerate(k.qubits):
                            if ind == 0:
                                noisy_circuit.append(cirq.PhaseFlipChannel(self.noise_strength*1e-3).on(q))
                                noisy_circuit.append(cirq.AmplitudeDampingChannel(self.noise_strength*1e-3).on(q))
                            else: #depolarizing acting only on target
                                noisy_circuit.append(cirq.DepolarizingChannel(depo_strength*self.noise_strength).on(q))
                                noisy_circuit.append(cirq.PhaseFlipChannel(self.noise_strength*1e-3).on(q))
                                noisy_circuit.append(cirq.AmplitudeDampingChannel(self.noise_strength*1e-3).on(q))
                    else:
                        raise ValueError("Three qubit gate??")
                    noisy_circuit.append(k)

                ### Measurement error
                for k in circuit.all_qubits():
                    noisy_circuit.append(cirq.BitFlipChannel(p=self.noise_strength*1e-2).on(k))

            circuit = cirq.Circuit(noisy_circuit)
        return circuit, circuit_db

    def initialize(self,**kwargs):
        mode=kwargs.get("mode","x")
        if mode.lower()=="x":
            circuit_db = templates.x_layer(self)
        elif mode.lower()=="xz":
            circuit_db = database.concatenate_dbs([templates.x_layer(self), templates.z_layer(self)])
        elif mode.lower()=="u1":
            circuit_db = templates.u1_layer(self)
        elif mode[:-1].lower()=="hea-":
            circuit_db = database.concatenate_dbs([templates.hea_layer(self)]*int(mode[-1]))
        elif mode == "hea":
            circuit_db = templates.hea_layer(self)
        else:
            circuit_db = templates.u2_layer(self)
        qnode, circuit_db = self.give_circuit(circuit_db)
        return circuit_db

    def draw(self, circuit_db):
        from utilities.translator.pennylane_translator import PennyLaneTranslator
        self.penny_translator = PennyLaneTranslator(self.n_qubits)
        return self.penny_translator.draw(circuit_db)
# class PennyLaneTranslator:
#     def __init__(self, n_qubits, **kwargs):
#         self.n_qubits = n_qubits
#         #self.qubits = cirq.GridQubit.rect(1, n_qubits)
#
#         ### blocks that are fixed-structure (i.e. channels, state_preparation, etc.)
#         untouchable_blocks = kwargs.get("untouchable_blocks",[])
#         self.untouchable_blocks = untouchable_blocks
#         self.discard_qubits = kwargs.get("discard_qubits",[]) ###these are the qubits that you don't measure, i.e. environment
#         self.device_name = kwargs.get("device_name","default.qubit")
#
#         #### keep a register on which integers corresponds to which CNOTS, target or control.
#         self.indexed_cnots = {}
#         self.cnots_index = {}
#         count = 0
#         for control in range(self.n_qubits):
#             for target in range(self.n_qubits):
#                 if control != target:
#                     self.indexed_cnots[str(count)] = [control, target]
#                     self.cnots_index[str([control,target])] = count
#                     count += 1
#         self.number_of_cnots = len(self.indexed_cnots)
#         #
#         self.cgates = {0:qml.RZ, 1: qml.RX, 2:qml.RY, 3:qml.Hadamard}
#         self.temp_circuit_db = {}
#
#         self.initialize(mode=kwargs.get("initialize","u1"))
#     def spit_gate(self,gate_id):
#         ## the symbols are all elements we added but the very last one (added on the very previous line)
#         ind = gate_id["ind"]
#
#         if ind<self.number_of_cnots:
#             qml.CNOT(self.indexed_cnots[str(ind)])
#         else:
#             cgate_type = (ind-self.number_of_cnots)%3
#             qubit = (ind-self.number_of_cnots)//self.n_qubits
#             symbol_name = gate_id["symbol"]
#             param_value = gate_id["param_value"]
#             self.cgates[cgate_type](param_value,qubit)
#
#     def append_to_circuit(self,circuit_db, gate_id):
#         ## the symbols are all elements we added but the very last one (added on the very previous line)
#         symbols = []
#         for j in [k["symbol"] for k in circuit_db.values()]:
#             if j != None:
#                 symbols.append(j)
#         ind = gate_id["ind"]
#
#         gate_index = len(list(circuit_db.keys()))
#         circuit_db[gate_index] = gate_id #this is the new item to add
#
#         if ind<self.number_of_cnots:
#             qml.CNOT(self.indexed_cnots[str(ind)])
#         else:
#             cgate_type = (ind-self.number_of_cnots)//self.n_qubits
#             qubit = (ind-self.number_of_cnots)%self.n_qubits
#             symbol_name = gate_id["symbol"]
#             param_value = gate_id["param_value"]
#             if symbol_name is None:
#                 symbol_name = "th_"+str(len(symbols))
#                 circuit_db[gate_index]["symbol"] = symbol_name
#             else:
#                 if symbol_name in symbols:
#                     print("Symbol repeated {}, {}".format(symbol_name, symbols))
#             if cgate_type < 3:
#                 self.cgates[cgate_type](param_value,qubit)
#             else:
#                 self.cgates[cgate_type](qubit)
#
#
#         return circuit_db
#
#
#
#     def give_circuit(self, dd,**kwargs):
#         """
#
#         """
#         unresolved = kwargs.get("unresolved",True)
#         just_call = kwargs.get("just_call",False)
#
#         dev = qml.device(self.device_name, wires=self.n_qubits)#, simulator=cirq.Simulator())
#         ### TO-DO: CHECK INPUT COPY!
#         @qml.qnode(dev)
#         def qnode(inputs, weights,**kwargs):
#             """
#             weights is a list of variables (automatic in penny-lane, here i feed [] so i don't update parameter values)
#             """
#
#             self.db = {}
#             cinputs = inputs#.copy()
#             symbols = database.get_trainable_symbols(self,cinputs)
#             ww = {s:w for s,w in zip( symbols, weights)}
#             cinputs = database.update_circuit_db_param_values(self, cinputs, ww)
#
#             list_of_gate_ids = [templates.gate_template(**dict(cinputs.iloc[k])) for k in range(len(cinputs))]
#             for i,gate_id in enumerate(list_of_gate_ids):
#                 self.db = self.append_to_circuit(self.db, gate_id)
#             return [qml.expval(qml.PauliZ(k)) for k in range(self.n_qubits)]
#
#         #self.db = {}
#         circuit = qnode(dd, []) ##this creates the database; the weights are used as trainable variables # of optimization
#         circuit_db = pd.DataFrame.from_dict(self.db,orient="index")
#         if just_call == False:
#             self.db = circuit_db.copy()
#             self.db_train = self.db.copy() ### copy to be used in PennyLaneModel
#         return qnode, circuit_db
#
#     def initialize(self,**kwargs):
#         mode=kwargs.get("mode","x")
#         if mode=="x":
#             circuit_db = templates.x_layer(self)
#         elif mode=="u1":
#             circuit_db = templates.u1_layer(self)
#         else:
#             circuit_db = templates.u2_layer(self)
#         qnode, circuit_db = self.give_circuit(circuit_db)
#         return circuit_db
#
#     def draw(self, circuit_db):
#         circuit, circuit_db = self.give_circuit(circuit_db, just_call=True)
#         return print(qml.draw(circuit)(circuit_db, []))
