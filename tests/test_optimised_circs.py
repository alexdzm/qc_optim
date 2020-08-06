import os
import time
import numpy as np
import qcoptim as qc
import joblib as jbl
import matplotlib.pyplot as plt

pi = np.pi
# ------------------------------------------------------
# General Qiskit related helper functions
# ------------------------------------------------------

if 'h3_circs_prmBool.dmp' not in os.listdir():
    raise FileNotFoundError("This test assumes you have circ dmp file: 'h3_circs_prmBool.dmp'")
data = jbl.load('h3_circs_prmBool.dmp')
inst = qc.utilities.quick_instance()

scf_energy = np.zeros(100)
cir_energy = np.zeros(100)
cost_list, wpo_list = [], []
for ii, cc in enumerate(data):
    qasm_str = cc['qasm']
    bool_lst = cc['should_prm']
    coords = cc['prm']
    wpo_list.append(qc.utilities.get_H_chain_qubit_op(coords))

    
    atom = 'H 0 0 0; H 0 0 {}; H 0 0 {}'.format(*np.cumsum(coords))
    ansatz = qc.ansatz.AnsatzFromQasm(qasm_str, bool_lst)

    cst = qc.cost.ChemistryCost(atom, ansatz, inst, verbose=False)
    scf_energy[ii] = cst._min_energy
    cir_energy[ii] = np.squeeze(cst(ansatz._x_sol))
    
    print('Min energy from pySCF   : ' + str(scf_energy[ii]))
    print('Min energy from our circ: ' + str(cir_energy[ii]))
    print('-----------')
    
plt_scf = np.reshape(scf_energy, (10, 10))
plt_cir = np.reshape(cir_energy, (10, 10))    
rescale = lambda x: (np.log(x +2))
f , ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
im = ax[0].pcolor(rescale(plt_scf))
ax[0].set_title('scf energies (log scale)')
ax[0].set_aspect('equal')
f.colorbar(im, ax=ax[0])


im = ax[1].pcolor(rescale(plt_cir))
ax[1].set_title('circuit energies (log scale)')
ax[1].set_aspect('equal')
f.colorbar(im, ax=ax[1])

wpo_list = qc.utilities.enforce_qubit_op_consistency(wpo_list)


cost_list = [qc.cost.CostWPO(ansatz, inst, ww) for ww in wpo_list]

domain = np.array([(0, 2*pi) for i in range(cost_list[0].ansatz.nb_params)])
bo_args = qc.utilities.gen_default_argsbo(f=lambda: 0.5,
                                          domain=domain,
                                          nb_init=33,
                                          eval_init=False)
bo_args['nb_iter'] = 15 


runner = qc.optimisers.ParallelRunner(cost_list, 
                                      qc.optimisers.MethodBO,
                                      optimizer_args = bo_args, 
                                      share_init = True, 
                                      method = '2d')


runner.next_evaluation_circuits()
print('there are {} init circuits'.format(len(runner.circs_to_exec)))

t = time.time()
bat = qc.utilities.Batch()
bat.submit_exec_res(runner)
print('took {:2g} s to run inits'.format(time.time() - t))

t = time.time()
runner.init_optimisers()
print('took {:2g} s to init the 100 optims from 33 points'.format(time.time() - t))

for ii in range(bo_args['nb_iter']):
    t = time.time()
    runner.next_evaluation_circuits()
    bat.submit_exec_res(runner)
    print('took {:2g} s to run circs'.format(time.time()  - t))
    t = time.time()
    runner.update()
    print('took {:2g} s to run {}th iter'.format(time.time() - t, ii))


x_opt_pred = [opt.best_x for opt in runner.optim_list]
runner.shot_noise(x_opt_pred, nb_trials=1)
bat.submit_exec_res(runner)
opt_energies = runner._results_from_last_x()
opt_energies_mat = np.reshape(np.squeeze(opt_energies), (10, 10))    


f , ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
im = ax[0].pcolor(rescale(plt_cir))
ax[0].set_title('circuit energies (log scale)')
ax[0].set_aspect('equal')
ax[0].set_ylabel('x3-x2')
ax[0].set_xlabel('x2-x1')
f.colorbar(im, ax=ax[0])


im = ax[1].pcolor(rescale(opt_energies_mat))
ax[1].set_title('2d NN BO opt: 33 init, 15 iter')
ax[1].set_aspect('equal')
ax[1].set_ylabel('x3-x2')
ax[1].set_xlabel('x2-x1')
f.colorbar(im, ax=ax[1])

import dill
with open('h3_chain_data.pkl', 'wb') as f:
    data = {'scf':plt_scf,
            'cir':plt_cir,
            'opt':opt_energies_mat}
    dill.dump(data)
    
    