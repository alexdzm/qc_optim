#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:10:36 2020

@author: fred
re-write test_GHZ with the new code
+ added pipeline for BEM
"""
import qiskit as qk
import numpy as np
import sys
sys.path.insert(0, '../core/')
import utilities as ut
import cost as cost
ut.add_path_GPyOpt()
    
import GPyOpt
import copy
import defined_circuits



#%% 
NB_SHOTS_DEFAULT = 8192
OPTIMIZATION_LEVEL_DEFAULT = 2
SINGAPORE_GATE_MAP_CYC_6 = [1,2,3,8,7,6] # Maybe put this in bem
SINGAPORE_GATE_MAP_CYC_6_EXTENDED = [2, 6, 10, 12, 14, 8] # Maybe put this in bem
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx0 = [1,3,2]
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx1 = [0,3,2]
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx2 = [0,4,2]
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx3 = [0,6,2]    
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx4 = [5,6,2] # might actually be 5 dheck/rerun
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx6 = [9,13,2] # might actually be 7 dheck/rerun


# ===================
# Choose a backend using the custom backend manager and generate an instance
# ===================

try:
    bem
except:
    bem = ut.BackendManager()
    
bem.get_current_status()
chosen_device = int(input('SELECT IBM DEVICE:'))
bem.get_backend(chosen_device, inplace=True)
inst = bem.gen_instance_from_current(nb_shots=NB_SHOTS_DEFAULT, 
                                     optim_lvl=OPTIMIZATION_LEVEL_DEFAULT,
                                     initial_layout=SINGAPORE_GATE_MAP_CYC_6_EXTENDED)

inst_ghz = bem.gen_instance_from_current(nb_shots=NB_SHOTS_DEFAULT, 
                                         optim_lvl=OPTIMIZATION_LEVEL_DEFAULT,
                                         initial_layout=ROCHESTER_GATE_MAP_GHZ_3_SWAPSx6)
                          


ansatz, nb_p, nb_q, x_sol = defined_circuits.GHZ_3qubits_6_params()
# different cost functions 


cost_cost = cost.GHZPauliCost(ansatz=ansatz, N=nb_q, instance=inst_ghz, nb_params=nb_p)

# cost_cost = cost.GraphCyclPauliCost(ansatz=ansatz, N=nb_q, instance=inst, nb_params=nb_p)

if bem.current_backend.name() != 'qasm_simulator':
    ansatz_transpiled = copy.deepcopy(cost_cost.main_circuit[0])
    print(cost_cost.check_depth(long_output=True))
    print(cost_cost.check_depth())
    print(cost_cost.check_layout())
    print(cost_cost.main_circuit)



keep_going = str(input('Everything_look okay?'))
if keep_going != 'y':
    this_will_throw_and_error
    
if x_sol is not None and bem.current_backend.name() == 'qasm_simulator':
    assert cost_cost(x_sol) == 1., "pb with ansatz/x_sol"
    # assert cost1(x_sol) == 1., "pb with ansatz/x_sol"
    # assert cost2(x_sol) == 1., "pb with ansatz/x_sol"
    


# ===================
# BO Optim
# Cost function defined by cost_cost
# ===================
# setu
    
print('''Warning: this assumes cost_cost is the default cost_function from here on''')
NB_INIT = 80
NB_ITER = 80
DOMAIN_FULL = [(0, 2*np.pi) for i in range(nb_p)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
cost_bo = lambda x: 1-cost_cost(x) 
bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN_FULL, nb_init=NB_INIT)
bo_args.update({'acquisition_weight': 7}) # increase exploration

#optim
Bopt = GPyOpt.methods.BayesianOptimization(**bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
#fid_test(x_seen)
#fid_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()





## ===================
## Get a baseline to compare to BO and save result
## ===================

x_opt_pred = Bopt.X[np.argmin(Bopt.model.predict(Bopt.X, with_noise=False)[0])]

if type(x_sol) != type(None):
    baseline_values = cost_cost.shot_noise(x_sol, 10)
else:
    baseline_values = None
bopt_values = cost_cost.shot_noise(x_opt_pred, 10)


ut.gen_pkl_file(cost_cost, Bopt, 
                baseline_values = baseline_values, 
                bopt_values = bopt_values, 
                dict_in = {'bo_args':bo_args})


