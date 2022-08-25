import pickle
from utilities.translator.pennylane_translator import PennyLaneTranslator
import numpy as np
import os
from datetime import datetime
from utilities.evaluator.misc import get_def_path

class PennyLaneEvaluator(PennyLaneTranslator):
    def __init__(self,minimizer, killer, args,
                **kwargs):
        """
        This class evaluates the cost at each iteration, and decides whether to accept the new circuit or not.

        It also stores the results either if there's a relevant modification or not.

        Finally, it allows for the possibilty of loading previous results.

        *** args = {"problem_name":str, params":list}
        *** acceptance_percentage: up to which value an increase in relative energy is accepted or not
        *** path:
            get_def_path() or not

        *** stopping criteria: relative error you will to accept.
        """
        super(PennyLaneEvaluator, self).__init__(n_qubits=args["n_qubits"])

        self.minimizer = minimizer
        self.killer = killer

        self.raw_history = {}
        self.evolution = {}
        self.displaying={"information":"\n VAns started at {} \n".format(datetime.now())}

        self.lowest_cost = None
        self.lower_bound = kwargs.get("lower_bound", -np.inf)

        args["params"] = np.round(args["params"],2)
        self.args = args
        if minimizer.noisy == True:
            self.identifier =  get_def_path() + "noisy_{}_{}/{}/{}/".format(minimizer.translator.noise_strength,args["problem"],args["params"], args["nrun"])
        else:
            self.identifier =  get_def_path() + "{}/{}/{}/".format(args["problem"],args["params"], args["nrun"])

        os.makedirs(self.identifier, exist_ok=True)

        self.lowest_acceptance_percentage = kwargs.get("lowest_acceptance_percentage", 1e-4)
        self.vans_its = kwargs.get("vans_its", 100)
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

    def accept_cost(self, C, circuit_db):
        """
        C: cost after some optimization (to be accepted or not).
        """
        if self.lower_bound == -np.inf:
            stop = False
        else:
            stop = (C - self.lower_bound)/np.abs(self.lower_bound) <= self.acceptance_percentage ## relative error up to 1e-2, i finish

        if self.lowest_cost is None: ###accept initial modification
            accept = True
        else:
            accept = (C-self.lowest_cost)/np.abs(self.lowest_cost) <= self.acceptance_percentage
        if accept == True:
            # print(accept, (C-self.lowest_cost)/np.abs(self.lowest_cost), C, self.lowest_cost)
            returned_db = circuit_db.copy()
            self.acceptance_percentage*=0.9
            self.killer.accept_wall=2/self.acceptance_percentage
        else:
            if self.its_without_improving > self.get_back_after_its:
                best_costs = [self.evolution[k][1] for k in range(len(list(self.evolution.keys())))]
                indi_optimal = np.argmin(best_costs)
                returned_db = self.evolution[indi_optimal][0]
                print("getting back to {}w/ cost {}".format(indi_optimal, best_costs[indi_optimal]))
                self.its_without_improving = 0
                #self.acceptange_percentage = max(1e-4self.initial_acceptange_percentage*(0.9**(len(self.evolution.keys())))
            else:
                self.its_without_improving+=1
                returned_db = circuit_db.copy()
                #self.acceptance_percentage*=10#
                #self.acceptance_percentage = min(1e-2, self.acceptance_percentage)
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
            # self.decrease_acceptance_range()
        else:
            self.its_without_improving+=1
            # if self.its_without_improving > int(self.get_back_after_its/2):
            #     self.increase_acceptance_range()

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
