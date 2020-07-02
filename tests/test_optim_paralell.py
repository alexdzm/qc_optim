#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:10:36 2020
@author: Kiran
"""

import sys
import copy
import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
sys.path.insert(0, '../qcoptim/')
import qcoptim.ansatz as az
import qcoptim.cost as cost
import qcoptim.utilities as ut
import qcoptim.optimisers as op

def add_reduced_domain(params, delta, domain):
    domain = list(domain)
    reduced = [[-delta, delta]]*params
    return np.array(domain+reduced)


# ===================
# Defaults
# ===================
pi= np.pi
NB_SHOTS_DEFAULT = 2048
OPTIMIZATION_LEVEL_DEFAULT = 0
TRANSPILER_SEED_DEFAULT = 10
NB_INIT = 100
NB_ITER = 50
NB_SWAPS = 0
NB_DELTA = pi/8
CHOOSE_DEVICE = True
SAVE_DATA = True


# ===================
# Choose a backend using the custom backend manager and generate an instance
# ===================
if CHOOSE_DEVICE:
    bem = ut.BackendManager()
    bem.get_current_status()
    chosen_device = int(input('SELECT IBM DEVICE:'))
    bem.get_backend(chosen_device, inplace=True)
    inst = bem.gen_instance_from_current(initial_layout=[1,3,2],
                                         nb_shots=NB_SHOTS_DEFAULT,
                                         optim_lvl=OPTIMIZATION_LEVEL_DEFAULT)
                                         #measurement_error_mitigation_cls=qk.ignis.mitigation.measurement.CompleteMeasFitter)


# ===================
# Generate ansatz and const functins (will generalize this in next update)
# ===================
x_sol = np.pi/2 * np.array([1.,1.,2.,1.,1.,1.])
x_sol_u3_2 = np.array([1.,  2.,  1.,  0.,  3.,  1.,  
                       2.,  3.,  1.,  2.,  2.,  3.,  
                       2.,  2.,  0., -0.,  2.,  3.])*pi/2
funcs = [az._GHZ_3qubits_6_params_cx0,
         az._GHZ_3qubits_6_params_cx1,
         az._GHZ_3qubits_6_params_cx2,
         az._GHZ_3qubits_6_params_cx3,
         az._GHZ_3qubits_6_params_cx4,
         az._GHZ_3qubits_6_params_cx5,
         az._GHZ_3qubits_6_params_cx6,
         az._GHZ_3qubits_6_params_cx7,
         az._GHZ_3qubits_cx7_u3_correction]
anz_vec = [az.AnsatzFromFunction(fun) for fun in funcs]
anz_vec = [az.RegularU3Ansatz(3, 1)]
cost_list = [cost.GHZPauliCost3qubits(anz, inst, invert=True) for anz in anz_vec]
nb_params = anz_vec[0].nb_params
cost_list = np.atleast_1d(cost_list[NB_SWAPS])


# ======================== /
#  Default BO optim args
# ======================== /
domain = [(0, 2*np.pi) for i in range(nb_params)]
domain = [x_sol_u3_2 - NB_DELTA*np.ones(nb_params), x_sol_u3_2 +  NB_DELTA*np.ones(nb_params)]
domain = np.array(domain).transpose()
if NB_SWAPS == -1:
    domain = add_reduced_domain(6, 0.1, domain)
    x_sol = list(x_sol) + [0]*6
bo_args = ut.gen_default_argsbo(f=lambda x: .5, 
                                domain=domain, 
                                nb_init=NB_INIT,
                                eval_init=False)

bo_args['nb_iter'] = NB_ITER + NB_INIT
bo_args['acquisition_weight'] = 2

spsa_args = {'a':1, 'b':0.628, 's':0.602, 
             't':0.101,'A':0,'domain':[(0, 2*np.pi) for i in range(anz_vec[0].nb_params)],
             'x_init':None}

# ======================== /
# Init runner classes with different methods
# ======================== /

opt_bo = op.MethodBO
opt_spsa = op.MethodSPSA

bo_args_list = [copy.deepcopy(bo_args) for ii in range(len(cost_list))]

runner1 = op.ParallelRunner(cost_list, 
                            opt_bo, 
                            optimizer_args = bo_args_list,
                            share_init = True,
                            method = 'independent')

# runner2 = op.ParallelRunner(cost_list[:2], 
#                             opt_spsa,
#                             optimizer_args = spsa_args,
#                             share_init = False,
#                             method = 'independent')

# runner3 = op.ParallelRunner(cost_list[:4], 
#                             [opt_bo],
#                             optimizer_args = bo_args,
#                             share_init = False,
#                             method = 'right')


# single_bo = op.SingleBO(cst0, bo_args)

# single_SPSA = op.SingleSPSA(cst0, spsa_args)

runner = runner1

x_new = [[x_sol, x_sol], [x_sol]]

# ========================= /
# Testing circ generation and output formainting
# ========================= /
if len(runner.optim_list) == 2:
    Batch = ut.Batch(inst)
    runner.next_evaluation_circuits()
    print(runner.method)
    print(runner._last_x_new)
    print('---------------------------------------------')
    runner._gen_circuits_from_params(x_new, inplace=True)
    print(runner._last_x_new)
    Batch.submit_exec_res(runner)
    print(runner._results_from_last_x())

# # ========================= /
# # And initilization:
# ========================= /
Batch = ut.Batch(inst)
runner.next_evaluation_circuits()
print(len(runner.circs_to_exec))
Batch.submit(runner)
Batch.execute()
Batch.result(runner)
runner.init_optimisers()

# optimizers now have new init info. 
try:
    print(runner.optim_list[0].optimiser.X)
    print(runner.optim_list[0].optimiser.Y)
except:
    pass

# Run optimizer step by step
for ii in range(NB_ITER):
    runner.next_evaluation_circuits()
    Batch.submit_exec_res(runner)
    runner.update()
    try:
        print(len(runner.optim_list[0]._x_mp))
    except:
        print(len(runner.optim_list[0].optimiser.X))


try: 
    for opt in runner.optim_list:
        bo = opt.optimiser
        bo.run_optimization(max_iter = 0, eps = 0) 
        (x_seen, y_seen), (x_exp,y_exp) = bo.get_best()
        print(bo.model.model)
        bo.plot_convergence()
        plt.show()
except:
    pass

# Get best_x
x_opt_pred = [opt.best_x for opt in runner.optim_list]

# Get baselines
if len(x_sol_u3_2) == anz_vec[0].nb_params:
    x_sol = x_sol_u3_2
    runner.shot_noise(x_sol, nb_trials=5)
    Batch.submit_exec_res(runner)
    baselines = runner._results_from_last_x()

# Get bopt_results
runner.shot_noise(x_opt_pred, nb_trials=5)
Batch.submit_exec_res(runner)
bopt_lines = runner._results_from_last_x()



# ======================== /
# Save BO's in different files
# ======================== /
if SAVE_DATA:
    for cst, bo, bl_val, bo_val in zip(runner.cost_objs,
                                       runner.optim_list,
                                       baselines,
                                       bopt_lines):
        bo = bo.optimiser
        bo_args = bo.kwargs
        ut.gen_pkl_file(cst, bo, 
                        baseline_values = bl_val, 
                        bopt_values = bo_val, 
                        info = 'SWP' + str(cst.main_circuit.depth()) + '_',
                        dict_in = {'bo_args':bo_args,
                                   'x_sol':x_sol})

#%% Everything here is old and broken

# # ======================== /
# #  Init each BO seperately (might put this in Batch class, or extended class)
# # ======================== /
# bo_arg_list, bo_list = [], []
# for cst in multi_cost.cost_list:
#     cost_bo = lambda x: 1-cst(x) 
#     bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN_FULL, nb_init=NB_INIT)
#     bo_args.update({'acquisition_weight': 7}) # increase exploration
#     bopt = GPyOpt.methods.BayesianOptimization(**bo_args)
#     bopt.run_optimization(max_iter = 0, eps = 0) 
    
#     # Opt runs on a list of bo args
#     bo_arg_list.append(bo_args)
#     bo_list.append(GPyOpt.methods.BayesianOptimization(**bo_args))
    
 
# # ======================== /
# #  Run opt using the nice efficient class (need to repackage)
# # ======================== /
# for ii in range(NB_ITER):
#     x_new = multi_cost.get_new_param_points(bo_list)
#     y_new = 1 - np.array(multi_cost(x_new))
#     multi_cost.update_bo_inplace(bo_list, x_new, y_new)
#     for bo in bo_list:
#         bo.acquisition.exploration_weight = dynamics_weight(ii)
#     print(ii)

 
# # ======================== /
# #  Print at results
# # ======================== /
# x_opt_pred = []
# for bo in bo_list:
#     bo.run_optimization(max_iter = 0, eps = 0) 
#     (x_seen, y_seen), (x_exp,y_exp) = bo.get_best()
#     #fid_test(x_seen)
#     #fid_test(x_exp)
#     print(bo.model.model)
#     bo.plot_convergence()
#     plt.show()
#     x_opt_pred.append(bo.X[np.argmin(bo.model.predict(bo.X, with_noise=False)[0])])


# # ======================== /
# # Get a baseline to compare to BO and save result
# # ======================== /
# if type(x_sol) != ut.NoneType:
#     baseline_values = multi_cost.shot_noise(x_sol, 10)
# else:
#     baseline_values = None
# bopt_values = multi_cost.shot_noise(x_opt_pred, 10)


# # ======================== /
# # Save BO's in different files
# # ======================== /
# for cst, bo, bl_val, bo_val, bo_args in zip(multi_cost.cost_list,
#                                             bo_list,
#                                             baseline_values,
#                                             bopt_values,
#                                             bo_arg_list):
#     ut.gen_pkl_file(cst, bo, 
#                     baseline_values = bl_val, 
#                     bopt_values = bo_val, 
#                     path = '/home/kiran/Documents/onedrive/Active_Research/QuantumSimulation/BayesianStateOptimizaiton/qc_optim/results_pkl/',
#                     info = 'cx' + str(cst.main_circuit.count_ops()['cx']) + '_',
#                     dict_in = {'bo_args':bo_args,
#                                'x_sol':x_sol})
    























# import qiskit
# from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
# from qiskit.providers.aer import noise # import AER noise model

# # Measurement error mitigation functions
# from qiskit.ignis.mitigation.measurement import (complete_meas_cal,
#                                                   CompleteMeasFitter, 
#                                                   MeasurementFilter)
# simulator = bem.current_backend

# # Generate a noise model for the qubits
# noise_model = noise.NoiseModel()
# for qi in range(5):
#     read_err = noise.errors.readout_error.ReadoutError([[0.75, 0.25],[0.1, 0.9]])
#     noise_model.add_readout_error(read_err, [qi])

# # Generate the measurement calibration circuits
# # for running measurement error mitigation
# qr = QuantumRegister(5)
# meas_cals, state_labels = complete_meas_cal(qubit_list=[2,3,4], qr=qr)

# # Execute the calibration circuits
# backend = qiskit.Aer.get_backend('qasm_simulator')
# job = qiskit.execute(meas_cals, backend=backend, shots=1000, noise_model=noise_model)
# cal_results = job.result()

# # Make a calibration matrix
# meas_fitter = CompleteMeasFitter(cal_results, state_labels)

# # Make a 3Q GHZ state
# cr = ClassicalRegister(3)
# ghz = QuantumCircuit(qr, cr)
# ghz.h(qr[2])
# ghz.cx(qr[2], qr[3])
# ghz.cx(qr[3], qr[4])
# ghz.measure(qr[2],cr[0])
# ghz.measure(qr[3],cr[1])
# ghz.measure(qr[4],cr[2])

# # Execute the GHZ circuit (with the same noise model)
# job = qiskit.execute(ghz, backend=backend, shots=1000, noise_model=noise_model)
# results = job.result()


# noise_model_ghz = noise.NoiseModel()
# for qi in range(3):
#     read_err = noise.errors.readout_error.ReadoutError([[0.75, 0.25],[0.1, 0.9]])
#     noise_model_ghz.add_readout_error(read_err, [qi])
# inst_corr = qk.aqua.quantum_instance.QuantumInstance(simulator, 
#                                                 measurement_error_mitigation_cls=CompleteMeasFitter,
#                                                 shots=1000,
#                                                 noise_model=noise_model_ghz)
# res_corr = inst_corr.execute(ghz)

# # Results without mitigation
# raw_counts = results.get_counts()
# print("Results without mitigation:", raw_counts)

# raw_corr_counts = res_corr.get_counts()
# print("Results with miticatin in inst:",raw_corr_counts)


# # Create a measurement filter from the calibration matrix
# meas_filter = meas_fitter.filter
# # Apply the filter to the raw counts to mitigate 
# # the measurement errors
# mitigated_counts = meas_filter.apply(raw_counts)
# print("Results with mitigation:", {l:int(mitigated_counts[l]) for l in mitigated_counts})