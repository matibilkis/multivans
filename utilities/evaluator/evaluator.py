import pickle
from utilities.translator.pennylane_translator import PennyLaneTranslator
import numpy as np
import os
from datetime import datetime
from utilities.evaluator.misc import get_def_path

class PennyLaneEvaluator(PennyLaneTranslator):
    def __init__(self,minimizer, killer, inserter, args,
                **kwargs):
        """
        This class evaluates the cost at each iteration, and decides whether to accept the new circuit or not.

        It also stores the results either if there's a relevant modification or not.

        Finally, it allows for the possibilty of loading previous results.

        *** args = {"problem_name":str, params":list}
        *** acceptance_percentage: up to which value an increase in relative energy is accepted or not
        *** path:
            get_def_path() or not

        *** stopping criteria: relative error you will to accept ---> notar this is accuracy to end now..
        """
        super(PennyLaneEvaluator, self).__init__(n_qubits=args["n_qubits"])

        self.minimizer = minimizer
        self.killer = killer
        self.inserter = inserter

        self.raw_history = {}
        self.evolution = {}
        self.displaying={"information":"\n VAns started at {} \n".format(datetime.now())}

        self.lowest_cost = None
        self.lower_bound = kwargs.get("lower_bound", -np.inf)

        args["params"] = np.round(args["params"],2)
        self.args = args
        if minimizer.noisy == True:
            if args["name"] == "":
                self.identifier =  get_def_path() + "{}_{}_{}_Q{}/{}/{}/".format(minimizer.noise_model, minimizer.translator.noise_strength,args["problem"],args["n_qubits"],args["params"], args["nrun"])
            else:
                self.identifier =  get_def_path() + "{}/{}_{}_{}_Q{}/{}/{}/".format(args["name"],minimizer.noise_model, minimizer.translator.noise_strength,args["problem"],args["n_qubits"],args["params"], args["nrun"])

        else:
            if args["name"] == "":
                self.identifier =  get_def_path() + "{}_Q{}/{}/{}/".format(args["problem"],args["n_qubits"],args["params"], args["nrun"])
            else:
                self.identifier =  get_def_path() + "{}/{}_Q{}/{}/{}/".format(args["name"],args["problem"],args["n_qubits"],args["params"], args["nrun"])

        os.makedirs(self.identifier, exist_ok=True)

        self.accuraccy_to_end = kwargs.get("accuraccy_to_end", 1e-4)
        self.lowest_acceptance_percentage = kwargs.get("lowest_acceptance_percentage", 1e-4)
        self.vans_its = kwargs.get("vans_its", 200)
        self.acceptance_percentage = kwargs.get("acceptance_percentage", 1e-2)
        self.get_back_after_its = kwargs.get("self.get_back_after_its",10)
        self.its_without_improving = 0

    def save_dicts_and_displaying(self):
        output = open(self.identifier+"/raw_history.pkl", "wb")
        pickle.dump(self.raw_history, output)
        output.close()
        output = open(self.identifier+"/evolution.pkl", "wb")
        pickle.dump(self.evolution, output)
        output.close()
        output = open(self.identifier+"/displaying.pkl", "wb")
        pickle.dump(self.displaying, output)
        output.close()
        return

    def load_dicts_and_displaying(self, folder, load_displaying=False):
        with open(folder+"raw_history.pkl" ,"rb") as h:
            self.raw_history = pickle.load(h)
        with open(folder+"evolution.pkl", "rb") as hh:
            self.evolution = pickle.load(hh)
        if load_displaying is True:
            with open(folder+"displaying.pkl", "rb") as hhh:
                self.displaying = pickle.load(hhh)
        return


    def increase_exploration(self):
        self.inserter.noise_in_rotations = min(.5, 2*self.inserter.noise_in_rotations)
        self.inserter.mutation_rate = min(1.75, 1.1*self.inserter.mutation_rate)
        self.inserter.prob_big=min(.05, 1.5*self.inserter.prob_big)
        self.inserter.p3body=min(.5, 1.5*self.inserter.p3body)
        self.minimizer.lr = max(0.1*self.accuraccy_to_end, 0.95*self.minimizer.lr)
        self.inserter.choose_qubit_Temperature = min(5*self.inserter.choose_qubit_Temperature, 100)
        self.inserter.pu1 = min(.1, self.inserter.pu1/2)

    def reset_exploration(self):
        self.inserter.mutation_rate = self.inserter.initial_mutation_rate
        self.inserter.noise_in_rotations = 0.1
        self.inserter.prob_big=self.inserter.initial_prob_big
        self.inserter.p3body=self.inserter.initial_p3body
        self.minimizer.lr = min(self.minimizer.initial_lr, 1.1*self.minimizer.lr)
        self.inserter.choose_qubit_Temperature = max(1, self.inserter.choose_qubit_Temperature/10)
        self.inserter.pu1 = self.inserter.initial_pu1

    def accept_cost(self, C, circuit_db):
        """
        C: cost after some optimization (to be accepted or not).
        """
        ### STOP vans or not
        if self.lower_bound == -np.inf:
            stop = False
        else:
            stop = (C - self.lower_bound)/np.abs(self.lower_bound) <= self.accuraccy_to_end

        ###accept initial modification always
        if self.lowest_cost is None:
            accept = True
        else:
            if C<self.lowest_cost:
                accept=True
                lowered = True
                self.minimizer.lr = max(0.1*self.accuraccy_to_end, 0.5*self.minimizer.lr)  #look finer grid
            else:
                relative_error = (C-self.lowest_cost)/np.abs(self.lowest_cost)
                accept = np.random.uniform() < np.exp(-np.abs(relative_error)/self.acceptance_percentage)
                lowered = False

        if (lowered==True):
            returned_db = circuit_db.copy()
            self.acceptance_percentage = max(1e-8, 0.9*self.acceptance_percentage)
            self.killer.accept_wall=2/self.acceptance_percentage
            self.reset_exploration()
            self.its_without_improving = 0
        else:
            self.its_without_improving+=1
            if self.its_without_improving >= self.get_back_after_its:
                accept = False
                best_costs = [self.evolution[k][1] for k in range(len(list(self.evolution.keys())))]
                indi_optimal = np.argmin(best_costs)
                returned_db = self.evolution[indi_optimal][0]
                print("getting back to {}w/ cost {}".format(indi_optimal, best_costs[indi_optimal]))
                self.its_without_improving = 0
                self.reset_exploration()
                self.acceptance_percentage = min(self.accuraccy_to_end, self.acceptance_percentage)
            else:
                self.its_without_improving+=1
                returned_db = circuit_db.copy()
                self.increase_exploration()
        return accept, stop, returned_db



    def add_step(self, database, cost,relevant=True, **kwargs):
        """
        database: pandas db encoding circuit
        cost: cost at current iteration
        relevant: if cost was decreased on that step as compared to previous one(s)
        """
        operation = kwargs.get("operation","variational")
        history = kwargs.get("history",[])

        if self.lowest_cost is None:
            self.lowest_cost = cost
            self.its_without_improving = 0

        elif cost < self.lowest_cost:
            self.lowest_cost = cost
            self.its_without_improving = 0
        else:
            self.its_without_improving+=1
        if self.lowest_cost <= self.lower_bound:
            self.end_vans = True
        self.raw_history[len(list(self.raw_history.keys()))] = [database, cost, self.lowest_cost, self.lower_bound, self.acceptance_percentage , operation, history]
        if relevant == True:
            self.evolution[len(list(self.evolution.keys()))] = [database, cost, self.lowest_cost, self.lower_bound, self.acceptance_percentage, operation, history]
        self.save_dicts_and_displaying()
        return


    def get_best_iteration(self):
        """
        returns minimum in evolution.
        """
        bests = list(np.where(np.array(list(self.evolution.values()))[:,1] == np.min(np.array(list(self.evolution.values()))[:,2]))[0])
        if len([np.squeeze(bests)])==0:
            return int(bests)
        else:
            return int(np.squeeze(bests[0]))

# def decrease_acceptance_range(self):
#     self.acceptance_percentage = max(self.lowest_acceptance_percentage, self.acceptance_percentage/self.acceptance_reduction_rate)
#     return
#
# def increase_acceptance_range(self):
#     self.acceptance_percentage = min(self.initial_acceptance_percentage, self.acceptance_percentage*self.acceptance_reduction_rate)
#     return
