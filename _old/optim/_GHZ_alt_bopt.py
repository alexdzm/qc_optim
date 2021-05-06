#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:10:36 2020

@author: fred
re-write test_GHZ with the new code
"""
import qiskit as qk
import numpy as np
import sys
sys.path.insert(0, '../core/')
import utilities as ut
import cost as cost
ut.add_path_GPyOpt()
import GPyOpt

NB_SHOTS_DEFAULT = 2048
OPTIMIZATION_LEVEL_DEFAULT = 3
# ===================
# Choose a backend using the custom backend manager and generate an instance
# ===================
bem = ut.BackendManager()
bem.get_current_status()
chosen_device = int(input('SELECT IBM DEVICE:'))
bem.get_backend(chosen_device, inplace=True)
inst = bem.gen_instance_from_current(nb_shots=NB_SHOTS_DEFAULT, 
                                     optim_lvl=OPTIMIZATION_LEVEL_DEFAULT)
inst_test = bem.gen_instance_from_current(nb_shots=8192, 
                                     optim_lvl=OPTIMIZATION_LEVEL_DEFAULT)

# ===================
# Define ansatz and initialize costfunction
# ===================
def ansatz(params):
    c = qk.QuantumCircuit(qk.QuantumRegister(1, 'a'), qk.QuantumRegister(1, 'b'), qk.QuantumRegister(1,'c'))
    c.rx(params[0], 0)
    c.rx(params[1], 1)
    c.ry(params[2], 2)
    c.barrier()
    c.cnot(0,2) 
    c.cnot(1,2) 
    c.barrier()
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    c.barrier()
    return c

# different cost functions 
ghz_fidelity = cost.GHZPauliCost(ansatz=ansatz, instance = inst, N=3, nb_params=6)
ghz_fidelity_test = cost.GHZPauliCost(ansatz=ansatz, instance = inst_test, N=3, nb_params=6)

# ===================
# BO Optim No noise / Use of fidelity / Normal workflow
# 20/25 works
# ===================
# setup
NB_INIT, NB_ITER = 30, 30
X_SOL = np.pi/2 * np.array([[1.,1.,2.,1.,1.,1.]])
DOMAIN = [(0, 2*np.pi) for i in range(6)]
cost_bo = lambda x: 1-ghz_fidelity(x) 
#now it evaluate the initial points
bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN, nb_init=NB_INIT)

#optim
Bopt = GPyOpt.methods.BayesianOptimization(**bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
ghz_fidelity_test(x_seen)
ghz_fidelity_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()




# ===================
# BO Optim No noise / Use of fidelity / Exposing the loop 
# 20/25 works
# use bo_args define in the previous example
# ===================
#init
Bopt = GPyOpt.methods.BayesianOptimization(**bo_args)   

#more init (may not be needed)
Bopt.run_optimization(max_iter = 0, eps = 0)

# This block is only to ensure the linear decrease of the exploration
if(getattr(Bopt, '_dynamic_weights') == 'linear'):
    update_weights = True
    def dynamics_weight(n_iter):
        return max(0.000001, bo_args['acquisition_weight'] * (1 - n_iter / NB_ITER))
else:
    update_weights = False

# Main loop
for iter_idx in range(NB_ITER):
    Bopt._update_model(Bopt.normalization_type) # to verif

    if(update_weights):
        Bopt.acquisition.exploration_weight = dynamics_weight(iter_idx)
    
    #suggested parameters
    Xnew = Bopt._compute_next_evaluations()
    
    #### EVAL FOR THE NEW SUGGESTION <- where you would do extra stuff
    Ynew = cost_bo(Xnew)

    #Incorporate new data Augment X
    Bopt.X = np.vstack((Bopt.X, Xnew))
    Bopt.Y = np.vstack((Bopt.Y, Ynew))

#finalize (may not be needed)
Bopt.run_optimization(max_iter = 0, eps = 0)


# Look at res
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
ghz_fidelity_test(x_seen)
ghz_fidelity_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()
