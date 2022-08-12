import tensorflow as tf
from pennylane import numpy as np
import pennylane as qml
import pandas as pd
import utilities.templates as templates


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

    def append_to_circuit(self,gate_id):
        ## the symbols are all elements we added but the very last one (added on the very previous line)
        symbols = []
        for j in [k["symbol"] for k in self.circuit_db.values()]:
            if j != None:
                symbols.append(j)

        ind = gate_id["ind"]

        gate_index = len(list(self.circuit_db.keys()))
        self.circuit_db[gate_index] = gate_id #this is the new item to add

        if ind<self.number_of_cnots:
            qml.CNOT(self.indexed_cnots[str(ind)])
        else:
            cgate_type = (ind-self.number_of_cnots)%3
            qubit = (ind-self.number_of_cnots)//self.n_qubits
            symbol_name = gate_id["symbol"]
            param_value = gate_id["param_value"]
            if symbol_name is None:
                symbol_name = "th_"+str(len(symbols))
                self.circuit_db[gate_index]["symbol"] = symbol_name
            else:
                if symbol_name in symbols:
                    print("warning, repeated symbol while constructing the circuit, see circuut_\n  symbol_name {}\n symbols {}\ncircuit_db {} \n\n\n".format(symbol_name, symbols, self.circuit_db))
            self.cgates[cgate_type](param_value,qubit)



    def give_circuit(self, dd,**kwargs):
        """
        retrieves circuit from circuit_db. It is assumed that the order in which the symbols are labeled corresponds to order in which their gates are applied in the circuit.
        If unresolved is False, the circuit is retrieved with the values of rotations (not by default, since we feed this to a TFQ model)
        """
        unresolved = kwargs.get("unresolved",True)


        dev = qml.device("default.qubit", wires=self.n_qubits)
        @qml.qnode(dev)
        def qnode(inputs, weights):
            circuit, self.circuit_db = [],{}
            list_of_gate_ids = [templates.gate_template(**dict(inputs.iloc[k])) for k in range(len(inputs))]
            for gate_id in list_of_gate_ids:
                self.append_to_circuit(gate_id)
            return [qml.expval(qml.PauliZ(k)) for k in range(self.n_qubits)]

        circuit = qnode(dd, 2)
        circuit_db = pd.DataFrame.from_dict(self.circuit_db,orient="index")
        return qnode, circuit_db







class modelito(tf.keras.Model):
    def __init__(self, qnode):
        super(modelito,self).__init__()

        weight_shapes = {"weights": qnode.train_params}
        self.qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=len(qnode.device.wires))

        self.cost_value = Metrica(name="cost")
        self.lr_value = Metrica(name="lr")
        self.gradient_norm = Metrica(name="grad_norm")

    @property
    def metrics(self):
        return [self.cost_value, self.lr_value,self.gradient_norm]

    def call(self, inputs):
        return self.qlayer(inputs)

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            cost = tf.math.reduce_sum(self(x))
        grads=tape.gradient(cost, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        gnorm = tf.reduce_sum(tf.pow(grads[0],2))#,)

        self.gradient_norm.update_state(tf.cast(gnorm, tf.dtypes.DType(1)))
        self.cost_value.update_state(tf.cast(cost, tf.dtypes.DType(1)))
        self.lr_value.update_state(tf.cast(self.optimizer.lr, tf.dtypes.DType(1)))
        return {k.name:k.result() for k in self.metrics}



class Metrica(tf.keras.metrics.Metric):
    def __init__(self, name):
        super(Metrica, self).__init__()
        self._name=name
        self.metric_variable = self.add_weight(name=name, initializer='zeros', dtype=np.float32)

    def update_state(self, new_value, sample_weight=None):
        self.metric_variable.assign(new_value)

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable.assign(0.)
