import numpy as np
import utilities.database.templates as templates
import utilities.database.database as database
from utilities.mutator.misc import gate_counter_on_qubits, get_symbol_number_from
from utilities.simplification.misc import get_qubits_involved, reindex_symbol, shift_symbols_up



class IdInserter:
    def __init__(self,
                translator,
                **kwargs):

        self.translator = translator
        n_qubits = self.n_qubits = translator.n_qubits
        self.spread_CNOTs=kwargs.get("spread_CNOTs",True)
        self.choose_qubit_Temperature = kwargs.get("choose_qubit_Temperature",10.)
        self.untouchable_blocks = kwargs.get("untouchable_blocks",[None])
        self.untouchable_qubits = kwargs.get("untouchable_qubits",[])
        self.noise_in_rotations=kwargs.get("noise_in_rotations",0.01)
        self.mutation_rate = kwargs.get("mutation_rate",1.5)
        self.initial_mutation_rate = kwargs.get("mutation_rate",1.5)
        self.prob_big = kwargs.get("prob_big",.1)
        self.p3body = kwargs.get("p3body",.1)
        self.initial_prob_big = kwargs.get("prob_big",.1)
        self.initial_p3body = kwargs.get("p3body",.1)
        self.touchable_qubits = list(range(n_qubits))

        for q in self.untouchable_qubits:
            self.touchable_qubits.remove(q)

        if isinstance(self.untouchable_blocks, int):
            self.untouchable_blocks = [self.untouchable_blocks]


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


    def resolution_1qubit(self, qubits):
        """
        retrieves rz rx rz on qubit q
        """
        q = qubits[0]
        rzq1 = self.number_of_cnots +  q
        rxq1 = self.number_of_cnots + self.n_qubits + q
        return [rzq1, rxq1, rzq1]

    def resolution_2cnots(self, qubits):
        """
        sequence of integers describing a CNOT, then unitary (compiled close to identity, rz rx rz) and the same CNOT
        q1: control qubit
        q2: target qubit
        """
        q1, q2 = qubits
        if q1==q2:
            raise Error("SAME QUBIT!")
        rzq1 = self.number_of_cnots + q1
        rzq2 = self.number_of_cnots +  q2
        rxq1 = self.number_of_cnots + self.n_qubits + q1
        rxq2 = self.number_of_cnots + self.n_qubits + q2
        cnot = self.cnots_index[str([q1,q2])] #q1 control q2 target
        if np.random.random() < self.prob_big:
            return [cnot, rzq1, rxq1, rzq1, rxq2, rzq2, rxq2, cnot]
        else:
            return [cnot, rxq1, rzq2, cnot]

    def resolution_3cnots(self, qubits):
        """
        sequence of integers describing a CNOT, then unitary (compiled close to identity, rz rx rz) and the same CNOT
        q1: control qubit
        q2: target qubit
        """
        q1, q2, q3 = qubits
        if (q1==q2) or (q1==q3) or(q2==q3):
            raise Error("SAME QUBIT!", qubits)
        rzq1 = self.number_of_cnots + q1
        rzq2 = self.number_of_cnots + q1
        rzq3 = self.number_of_cnots + q2
        rxq1 = self.number_of_cnots + self.n_qubits + q1
        rxq2 = self.number_of_cnots + self.n_qubits + q1
        rxq3 = self.number_of_cnots + self.n_qubits + q2
        cnot13 = self.cnots_index[str([q1,q3])] #q1 control q2 target
        cnot12 = self.cnots_index[str([q1,q2])] #q1 control q2 target
        if np.random.random() < self.prob_big:
            return [cnot13, rxq1, cnot12, rxq1, cnot12, rxq1, cnot13]
        else:
            return [cnot13, rxq1, rxq2, rxq3, cnot12, rxq1, rxq2, rxq3, cnot12, rxq1, rxq2, rxq3, cnot13]

    def mutate(self, circuit_db, cost, lowest_cost, T_spread=1e3):

        relative_energy = np.abs((cost - lowest_cost)/lowest_cost)
        if relative_energy < 1e-1:
           self.choose_qubit_Temperature = 1.
        else:
           self.choose_qubit_Temperature = T_spread

        ngates = np.random.exponential(scale=self.mutation_rate)
        nmutations = int(ngates+1)
        m_circuit_db = self.inserter(circuit_db)
        for ll in range(nmutations-1):
            m_circuit_db = self.inserter(m_circuit_db)
        return m_circuit_db, nmutations

    def inserter(self, circuit_db):
        """
        Inserts resolution of identity at some place in circuit_db.
        Takes into acconut density of gates wrt of entangling gates, active qubits.

        It also consider leaving untouched some block_id's (specified in untouchable_blocks)
        Returns a mutated circuit_db.
        """

        ### which qubit(s) will be touched by the new insertion ?
        ## count how many (rotations, CNOTS) acting on each qubit (rows)
        ngates = gate_counter_on_qubits(self,circuit_db, untouchable_qubits=self.untouchable_qubits)
        ngates_CNOT = ngates[:,1] ##number of CNOTs on each qubit
        qubits_not_CNOT = np.where(ngates_CNOT == 0)[0] ### target qubits are chosen later

        if (self.spread_CNOTs is True):
            prot = .1#prob_rot_cnot = [p, 1-p]
        else:
            prot = .5#[.5,.5]

        #### CHOOSE BLOCK #### 0--> rotation, 1 ---> CNOT
        which_block = np.random.choice([0,1,2], p=[prot, (1-self.p3body)*(1-prot), self.p3body*(1-prot)])#$which_prob(qubits_not_CNOT))

        if which_block == 0:
            gc=ngates[:,0]+1 #### gives the gate population for each qubit
            probs=np.exp(self.choose_qubit_Temperature*(1-gc/np.sum(gc)))/np.sum(np.exp(self.choose_qubit_Temperature*(1-gc/np.sum(gc))))
            qubits= np.random.choice(self.touchable_qubits,1,p=probs)
        else:
            if len(qubits_not_CNOT) > 1:
                self.choose_qubit_Temperature = 20
            else:
                self.choose_qubit_Temperature = 10

            gc=ngates[:,1]+1 #### gives the gate population for each qubit (of CNOTs!)
            probs=np.exp(self.choose_qubit_Temperature*(1-gc/np.sum(gc)))/np.sum(np.exp(self.choose_qubit_Temperature*(1-gc/np.sum(gc))))
            qubits = np.random.choice(self.touchable_qubits,which_block+1,p=probs,replace=False)

        ### this gives the list of gates to insert
        block_of_gates = [self.resolution_1qubit, self.resolution_2cnots, self.resolution_3cnots][which_block](qubits)

        ## position in the circuit to insert ?
        c1 = circuit_db[circuit_db["trainable"]==True]
        blocks = list(set(c1["block_id"]))
        for b in self.untouchable_blocks:
            if b in blocks:
                blocks.remove(b)
            which_circuit_block = np.random.choice(blocks, 1)[0]
        c2 = c1[c1["block_id"] == which_circuit_block]
        insertion_index = np.squeeze(np.random.choice(c2.index, 1))

        m_circuit_db =circuit_db.copy()

        for mind, m_gate in enumerate(block_of_gates):
            if m_gate < self.number_of_cnots:
                m_circuit_db.loc[insertion_index+0.1 + mind] = templates.gate_template(m_gate, block_id = which_circuit_block)
                m_circuit_db = m_circuit_db.sort_index().reset_index(drop=True)
            else:
                number_symbol_shifting = get_symbol_number_from(insertion_index+mind, m_circuit_db)
                m_circuit_db.loc[insertion_index+0.1 + mind] = templates.gate_template(m_gate, param_value=2*np.pi*np.random.random()*self.noise_in_rotations,
                                                                            symbol="th_"+str(number_symbol_shifting), block_id=which_circuit_block)
                m_circuit_db = m_circuit_db.sort_index().reset_index(drop=True)
                m_circuit_db = shift_symbols_up(self, insertion_index + mind, m_circuit_db)
        mcircuit, m_circuit_db = self.translator.give_circuit(m_circuit_db) ### this is because i save trainable db inside
        return m_circuit_db
