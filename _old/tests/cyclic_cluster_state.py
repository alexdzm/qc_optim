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
import qcoptim.ansatz as anz
import qcoptim.cost as cost
import qcoptim.utilities as ut
import qcoptim.optimisers as op
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter


# ===================
# Defaults
# ===================
pi= np.pi
NB_SHOTS_DEFAULT = 2**13
OPTIMIZATION_LEVEL_DEFAULT = 1
TRANSPILER_SEED_DEFAULT = 10
NB_INIT = 100
NB_ITER = 100
NB_SWAPS = 0
NB_DELTA = pi/24
CHOOSE_DEVICE = True
SAVE_DATA = False
SING_LAYOUT = [1, 2, 3, 8, 7, 6]
ROCH_LAYOUT = [0, 1, 2, 3, 4, 6, 13, 12, 11, 10, 9, 5]
ROCH_LAYOUT_6 = [0, 2, 4, 13, 11, 9]

# ===================
# Choose a backend using the custom backend manager and generate an instance
# ===================
if CHOOSE_DEVICE:
    bem = ut.BackendManager()
    bem.get_current_status()
    chosen_device = int(input('SELECT IBM DEVICE:'))
    if chosen_device == 1:
        initial_layout = ROCH_LAYOUT_6
        noise_model = None
    elif chosen_device == 3:
        initial_layout = SING_LAYOUT
        noise_model = None
    else:
        initial_layout = None
        noise_model = ut.gen_quick_noise()
    bem.get_backend(chosen_device, inplace=True)
    inst = bem.gen_instance_from_current(initial_layout=initial_layout,
                                         nb_shots=NB_SHOTS_DEFAULT,
                                         optim_lvl=OPTIMIZATION_LEVEL_DEFAULT,
                                         noise_model=noise_model,
                                         measurement_error_mitigation_cls=CompleteMeasFitter)


#%%
ans = anz.sandwitch_ansatzes.GHZ_plus_rotation(4, 'rx', 'rz')
cst = cost.StateFidelityCost('ghz', ans, inst, invert=True)
x_sol = [0]*ans.nb_params
nb_params = ans.nb_params


# ======================== /
#  Default BO optim args
# ======================== /
domain = np.array([(-NB_DELTA, NB_DELTA) for i in range(nb_params)])
domain += np.array([x_sol,x_sol]).transpose()
bo_args = ut.gen_default_argsbo(f=lambda x: .5,
                                domain=domain,
                                nb_init=NB_INIT,
                                eval_init=False)
bo_args['nb_iter'] = NB_ITER + NB_INIT
bo_args['acquisition_weight'] = 5
bo_args['acquisition_type'] = 'EI'
bo_args['initial_design_type'] = 'latin'
# ======================== /
# Init runner classes with different methods
# ======================== /
runner = op.ParallelRunner([cst],
                           op.MethodBO,
                           optimizer_args = bo_args,
                           share_init = True,
                           method = 'independent')


# ========================= /
# And initilization:
# ========================= /
Batch = ut.Batch(inst)
runner.next_evaluation_circuits()
print('Circs to init: {}'.format(len(runner.circs_to_exec)))
Batch.submit(runner)
Batch.execute()
runner._last_results_obj = Batch._last_results_obj
runner.init_optimisers()


# ========================= /
# Ensure optim sees expected solution:
# ========================= /
runner._gen_circuits_from_params([[x_sol]], inplace=True)
Batch.submit_exec_res(runner)
runner.update()


# ========================= /
# Run optimizer step by step
# ========================= /
for ii in range(NB_ITER):
    runner.next_evaluation_circuits()
    Batch.submit(runner)
    Batch.execute()
    runner._last_results_obj = Batch._last_results_obj
    runner.init_optimisers()    runner.update()
    print(len(runner.optim_list[0].optimiser.X))


for opt in runner.optim_list:
    bo = opt.optimiser
    bo.run_optimization(max_iter = 0, eps = 0)
    (x_seen, y_seen), (x_exp,y_exp) = bo.get_best()
    print(bo.model.model)
    bo.plot_convergence()
    plt.show()


# Get best_x
x_opt_pred = [opt.best_x for opt in runner.optim_list]

# Get baselines
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


