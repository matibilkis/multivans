import numpy as np
import pennylane as qml
# from utilities.circuit_database import CirqTranslater
import utilities.translator.pennylane_translator as penny_translator
import utilities.database.database as database
import utilities.database.templates as templates
from utilities.database.database import concatenate_dbs
from utilities.simplification.misc import get_qubits_involved, reindex_symbol, shift_symbols_down, shift_symbols_up, type_get, check_rot, order_symbol_labels, check_cnot, check_symbols_ordered, u2zxz
#from utilities.variational import Minimizer
#from utilities.compiling import *


class PennyLane_Simplifier:
    """
    untouchable::: list of blocks which simplifier should not toch (for instance environment blocks), state preparation blocks, fixed measurement blocks, etc.

    rule_1: CNOT when control is |0> == identity.    RELATIVE
    rule_2: Two same CNOTS -> identity (delete both).   ABSOLUTE
    rule_3: Rz (|0>) ---> kill (only adds a phase killed tracing over, in case we are computing).   RELATIVE
    rule_4:  Repeated rotations: add the values.    ABSOLUTE
    rule_5: compile 1-qubit gates into euler rotations.  ABSOLUTE
    rule_6: move cnots to the left, rotations to the right ABSOLUTE

    NOTE:
        we use pennylane to compile one-qubit unitary (we could do this by hand)
    """

    def __init__(self,translator, **kwargs):
        self.translator = translator
        self.max_cnt = kwargs.get('max_cnt',150)
        self.apply_relatives_to_first = kwargs.get("apply_relatives_to_first",True)
        self.absolute_rules = [self.rule_2, self.rule_4, self.rule_5,self.rule_6]  # self.rule_5# this are rules that should be applied to any block ,regardless of its position ..
        self.relative_rules = [self.rule_1, self.rule_3] ##rule 1 and 3 are not always applied since the incoming state is likely not |0> for general block_id's. (suppose you have a channel...)
        self.rules = self.relative_rules + self.absolute_rules
        self.loop_the_rules = 10 ### one could think on looping more than ones the rule
        self.apply_relatives_to_first = True
        self.untouchable = self.translator.untouchable_blocks ### for instance, channel blocks...

    def reduce_circuit(self, circuit_db):
        simplified_db = circuit_db.copy()
        nsimps=0
        for routine_check in range(100):
            final_cnt = 0
            ip, icnot = database.describe_circuit(self.translator, simplified_db)
            for rule in self.rules:
                cnt, simplified_db  = self.apply_rule(simplified_db  , rule)
                final_cnt += cnt
            simplified_db = self.order_symbols(simplified_db) ## this is because from block to block there might be a gap in the symbols order!
            sc, simplified_db = self.translator.give_circuit(simplified_db) ### and this is because I save current db to train inside translator..

            cp, ccnot = database.describe_circuit(self.translator,simplified_db)
            if not ((cp < ip) or (ccnot < icnot)):
                break

        return simplified_db, nsimps

    def apply_rule(self, original_circuit_db, rule, **kwargs):
        simplified, cnt = True, 0
        original_circuit, original_circuit_db = self.translator.give_circuit(original_circuit_db)
        gates_on_qubit, on_qubit_order = self.get_positional_dbs(original_circuit, original_circuit_db)
        simplified_db = original_circuit_db.copy()
        rules=[]
        while simplified and cnt < self.max_cnt:
            ss = simplified_db.copy()
            simplified, simplified_circuit_db = rule(simplified_db, on_qubit_order, gates_on_qubit)
            # if check_symbols_ordered(simplified_circuit_db) is False:
            #     simplified_db.to_csv("testing/data/dcl")
            #     ss.to_csv("testing/data/dcl_prev")
            #     raise AttributeError("ojo {}".format(rule))
            circuit, simplified_db = self.translator.give_circuit(simplified_circuit_db)
            gates_on_qubit, on_qubit_order = self.get_positional_dbs(circuit, simplified_db)
            # if simplified == True:
                # print("simplified using ",rule)
            cnt+=1
            if cnt>int(0.9*self.max_cnt):
                print("hey, i'm still simplifying, cnt{}".format(cnt))
                # print(rules)
        return cnt, simplified_db


    def get_positional_dbs(self, circuit, circuit_db):
        qubits_involved = get_qubits_involved(self.translator, circuit_db)

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

    def order_symbols(self, simplified_db):
        shift_need = True
        ssdb = simplified_db.copy()
        while shift_need is True:
            ss = ssdb["symbol"].dropna()
            prev_s = int(list(ss)[0].replace("th_",""))
            for ind,s in zip(ss.index[1:], ss[1:]):
                current = int(s.replace("th_",""))
                if current - prev_s >1:
                    shift_need = True
                    from_ind = ind
                    ssdb = shift_symbols_down(self.translator, from_ind, ssdb)
                    break
                else:
                    shift_need = False
                    prev_s = current
        return ssdb

    def rule_1(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        CNOT when control is |0> == identity.    RELATIVE
        """
        simplification = False

        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path):
                if ind_gate < self.translator.number_of_cnots:
                    control, target = self.translator.indexed_cnots[str(ind_gate)]
                    if (q == control) and (order_gate_on_qubit == 0) and (len(qubit_gates_path)>1):
                        pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]

                        block_id = simplified_db.loc[pos_gate_to_drop]["block_id"]
                        # simplified_db = shift_symbols_up(self.translator, int(pos_gate_to_drop), simplified_db)
                        # simplified_db = shift_symbols_up(self.translator, int(pos_gate_to_drop), simplified_db)
                        # simplified_db.loc[int(pos_gate_to_drop)+0.1] = gate_template(self.translator.number_of_cnots + self.translator.n_qubits + control, param_value=0.0, block_id=block_id)
                        # simplified_db.loc[int(pos_gate_to_drop)+0.11] = gate_template(self.translator.number_of_cnots + self.translator.n_qubits + target, param_value=0.0, block_id=block_id)
                        simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)

                        simplification = True
                        break
        simplified_db = simplified_db.sort_index().reset_index(drop=True)
        return simplification, simplified_db


    def rule_2(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        Two same CNOTS -> identity (delete both).   ABSOLUTE
        """
        simplification = False

        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):

                next_ind_gate = qubit_gates_path[order_gate_on_qubit+1]
                if (ind_gate < self.translator.number_of_cnots) and (ind_gate == next_ind_gate):
                    control, target = self.translator.indexed_cnots[str(ind_gate)]
                    not_gates_in_between = False
                    this_qubit = q
                    other_qubits = [control, target]
                    other_qubits.remove(q)
                    other_qubit = other_qubits[0]

                    ## now we need to check what happens in the other_qubit
                    for qord_other, ind_gate_other in enumerate(gates_on_qubit[other_qubit][:-1]):
                        if (ind_gate_other == ind_gate) and (gates_on_qubit[other_qubit][qord_other +1] == ind_gate):
                            ## if we append the CNOT for q and other_q on the same call, and also for the consecutive
                            ## note that in between there can be other calls for other qubits
                            order_call_q = on_qubit_order[q][order_gate_on_qubit]
                            order_call_other_q = on_qubit_order[other_qubit][qord_other]

                            order_call_qP1 = on_qubit_order[q][order_gate_on_qubit+1]
                            order_call_other_qP1 = on_qubit_order[other_qubit][qord_other+1]

                            ## then it's kosher to say they are consecutively applied (if only looking at the two qubits)
                            if (order_call_q == order_call_other_q) and (order_call_qP1 == order_call_other_qP1):

                                pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]
                                simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)
                                pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit+1]
                                simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)

                                simplification = True
                                break
                    if simplification is True:
                        break
        simplified_db = simplified_db.reset_index(drop=True)
        return simplification, simplified_db



    def rule_3(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        Rz (|0>) ---> kill (only adds a phase killed tracing over, in case we are computing).   RELATIVE
        """
        simplification = False
        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):
                if (order_gate_on_qubit == 0) and (self.translator.number_of_cnots <= ind_gate< self.translator.number_of_cnots+ self.translator.n_qubits ) and (len(qubit_gates_path[:-1])>1):
                    pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]
                    simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)
                    simplified_db = simplified_db.reset_index(drop=True)
                    simplified_db = shift_symbols_down(self.translator, pos_gate_to_drop, simplified_db)
                    simplification = True
                    break
        return simplification, simplified_db



    def rule_4(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        Repeated rotations: add the values.    ABSOLUTE
        """
        simplification = False
        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):
                if ind_gate>=self.translator.number_of_cnots:
                    next_ind_gate = qubit_gates_path[order_gate_on_qubit+1]
                    if next_ind_gate == ind_gate:
                        pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]
                        pos_gate_to_add = on_qubit_order[q][order_gate_on_qubit+1]

                        value_1 = simplified_db.loc[pos_gate_to_drop]["param_value"]
                        value_2 = simplified_db.loc[pos_gate_to_add]["param_value"]

                        ## i'm skeptical on this, would be better to access from param_value maybe.
                        simplified_db.loc[pos_gate_to_add] = simplified_db.loc[pos_gate_to_add].replace(to_replace=value_2, value=value_1 + value_2)
                        simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)
                        simplified_db = simplified_db.reset_index(drop=True)
                        simplified_db = shift_symbols_down(self.translator, pos_gate_to_drop, simplified_db)
                        simplified_db = self.order_symbols(simplified_db)
                        simplified_db = order_symbol_labels(simplified_db)
                        simplification = True

                        break
        return simplification, simplified_db




    def rule_5(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        compile 1-qubit gates into euler rotations.  ABSOLUTE
        """
        simplification = False
        original_db = simplified_db.copy() #this is not the most efficient thing, just a patch
        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-2]):

                if simplification is True:
                    break
                ind_gate_p1 = qubit_gates_path[order_gate_on_qubit+1]
                ind_gate_p2 = qubit_gates_path[order_gate_on_qubit+2]

                if (check_rot(ind_gate, self.translator) == True) and (check_rot(ind_gate_p1, self.translator) == True) and (check_rot(ind_gate_p2, self.translator) == True):

                    type_0 = type_get(ind_gate,self.translator)
                    type_1 = type_get(ind_gate_p1,self.translator)
                    type_2 = type_get(ind_gate_p2,self.translator)

                    if type_0 == type_2:
                        types = [type_0, type_1, type_2]
                        for next_order_gate_on_qubit, ind_gate_next in enumerate(qubit_gates_path[order_gate_on_qubit+3:]):
                            if (check_rot(ind_gate_next, self.translator) == True):# and (next_order_gate_on_qubit < len(qubit_gates_path[order_gate_on_qubit+3:])):
                                types.append(type_get(ind_gate_next, self.translator))
                                simplification=True
                            else:
                                break
                        if simplification == True:
                            indices_to_compile = [on_qubit_order[q][order_gate_on_qubit+k] for k in range(len(types))]
                            self.translator_ = penny_translator.PennyLaneTranslator(n_qubits=1)
                            u_to_compile_db = simplified_db.loc[indices_to_compile]
                            u_to_compile_db["ind"] = self.translator_.n_qubits*type_get(u_to_compile_db["ind"], self.translator) + self.translator_.number_of_cnots

                            devi, u_to_compile_db = self.translator_.give_circuit(u_to_compile_db)
                            target_u = qml.matrix(devi)(u_to_compile_db,[])

                            params = np.array(u2zxz(target_u))
                            u1s = templates.zxz_db(self.translator_, 0, params=True)
                            u1s["param_value"] = params

                            first_symbols = simplified_db["symbol"][indices_to_compile][:3]
                            for new_ind, typ, pval in zip(indices_to_compile[:3],[0,1,0], list(u1s["param_value"])):
                                simplified_db.loc[new_ind+0.1] = templates.gate_template(self.translator.number_of_cnots + q + typ*self.translator.n_qubits,
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
                            break
        return simplification, simplified_db

    def rule_6(self, simplified_db, on_qubit_order, gates_on_qubit):

        simplification = False
        if check_symbols_ordered(simplified_db) == False:
            raise AttributeError("pero cheee!!!!!")
        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):
                if simplification is True:
                    break
                ind_gate_p1 = qubit_gates_path[order_gate_on_qubit+1]
                ## if i have a rotation and then a CNOT
                if (check_rot(ind_gate, self.translator) is True) and (check_cnot(ind_gate_p1, self.translator) is True):
                    type_0 = type_get(ind_gate, self.translator)
                    control, target = self.translator.indexed_cnots[str(ind_gate_p1)]

                    this_qubit = q
                    other_qubits = [control, target]
                    other_qubits.remove(q)
                    other_qubit = other_qubits[0]

                    ### now it happens two interesting things: type0 == 0 AND q == control
                    ### or type_0 == 1 AND q == target  then swap orders
                    if ((type_0 == 0) and (q == control)) or ((type_0 == 1) and (q == target)):
                        if len(on_qubit_order[q]) <2:
                            simplification=False
                        else:
                            simplification = True
                            ###now we swap the order in which we apply the rotation and the CNOT.
                            index_rot = on_qubit_order[q][order_gate_on_qubit]
                            info_rot = simplified_db.loc[index_rot].copy()
                            simplified_db = simplified_db.drop(labels=[index_rot],axis=0)#

                            simplified_db.loc[on_qubit_order[q][order_gate_on_qubit+1 ] + 0.1] = info_rot
                            simplified_db = simplified_db.sort_index().reset_index(drop=True)
                            simplified_db = order_symbol_labels(simplified_db)
                            break
        return simplification, simplified_db
