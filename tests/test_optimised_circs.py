import os
import time
import numpy as np
import qcoptim as qc
import joblib as jbl
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
pi = np.pi
# ------------------------------------------------------
# General Qiskit related helper functions
# ------------------------------------------------------
method = '2d' # 'independent_plus_random_4'
nb_init = 10
nb_iter = 15
shape = (8, 8)
positions = np.linspace(0.2, 1.8, shape[0])


if 'h3_circs_prmBool.dmp' not in os.listdir():
    raise FileNotFoundError("This test assumes you have circ dmp file: 'h3_circs_prmBool.dmp'")
data = jbl.load('h3_circs_prmBool.dmp')
inst = qc.utilities.quick_instance()
ansatz = qc.ansatz.AnsatzFromQasm(data[0]['qasm'], data[0]['should_prm'])
rescale = lambda x: (np.log(x +2))

if False:
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
    
    
wpo_list = [qc.utilities.get_H_chain_qubit_op([dx1,dx2]) for dx1 in positions for dx2 in positions]
wpo_list = qc.utilities.enforce_qubit_op_consistency(wpo_list)

#%% Create cost functions
ansatz = qc.ansatz.AnsatzFromQasm(data[0]['qasm'], data[0]['should_prm'])
cost_list = [qc.cost.CostWPO(ansatz, inst, ww) for ww in wpo_list]
ed_energies_mat = [c._min_energy for c in cost_list]
ed_energies_mat = np.reshape(ed_energies_mat, shape)

domain = np.array([(0, 2*pi) for i in range(cost_list[0].ansatz.nb_params)])
bo_args = qc.utilities.gen_default_argsbo(f=lambda: 0.5,
                                          domain=domain,
                                          nb_init=nb_init,
                                          eval_init=False)
bo_args['nb_iter'] = nb_iter


runner = qc.optimisers.ParallelRunner(cost_list,
                                      qc.optimisers.MethodBO,
                                      optimizer_args = bo_args,
                                      share_init = True,
                                      method = method)


runner.next_evaluation_circuits()
print('there are {} init circuits'.format(len(runner.circs_to_exec)))

t = time.time()
bat = qc.utilities.Batch()
bat.submit_exec_res(runner)
print('took {:2g} s to run inits'.format(time.time() - t))

t = time.time()
runner.init_optimisers()
print('took {:2g} s to init the {} optims from {} points'.format(time.time() - t, shape[0]**2, bo_args['initial_design_numdata']))

for ii in range(bo_args['nb_iter']):
    t = time.time()
    runner.next_evaluation_circuits()
    print('took {:2g} s to optim acq function'.format(time.time()  - t))

    t = time.time()
    bat.submit_exec_res(runner)
    print('took {:2g} s to run circs'.format(time.time()  - t))
    
    t = time.time()
    runner.update()
    print('took {:2g} s to run {}th update'.format(time.time() - t, ii))


x_opt_pred = [opt.best_x for opt in runner.optim_list]
runner.shot_noise(x_opt_pred, nb_trials=1)
bat.submit_exec_res(runner)
opt_energies = runner._results_from_last_x()
opt_energies_mat = np.reshape(np.squeeze(opt_energies), shape)


f , ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
im = ax[0].pcolor(rescale(ed_energies_mat))
ax[0].set_title('ed energies (log scale)')
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





#%% Plotting y_ii - y_fin
# Set up fig stuff
f, ax = plt.subplots(3, 5, sharex=True, sharey=True)
ax = np.ravel(ax)


# Get optim list, and 'best' values of y (approx as last)
opt = runner.optim_list
y_last = np.squeeze([opt[ii].optimiser.Y[-1] for ii in range(shape[0]**2)]).reshape(*shape)
y_best = [min(opt[ii].optimiser.Y) for ii in range(shape[0]**2)]
y_best = np.reshape(y_best, shape)

# Gen NN mask (different iter for each point on grid)
x = np.ones(shape)
x = np.pad(x, [1,1], mode='constant', constant_values=[0,0])
y = [[0, 1, 0], [1,1, 1], [0, 1, 0]]
if '2d' in method:
    nn_mask = convolve2d(x, y, 'valid')
else:
    nn_mask = 5*np.ones(shape)

for ii in range(bo_args['nb_iter']):
    these_coords = bo_args['initial_design_numdata'] + ii*nn_mask
    these_coords = [int(cc) for cc in np.ravel(these_coords)]
    
    y_ii = [opt[ii].optimiser.Y[these_coords[ii]] for ii in range(shape[0]**2)]
    y_ii = np.squeeze(y_ii).reshape(*shape)
    
    diff = np.abs(y_ii - y_last)
    
    im = ax[ii].pcolor(diff, vmin=0.0,vmax=2)
    f.colorbar(im, ax=ax[ii])
    ax[ii].set_title('iter' + str(ii))
    
    if ii > 9:
        ax[ii].set_xlabel('x2-x1 (au)')
    if ii%5==0:
        ax[ii].set_ylabel('x3-x2 (au)')
ax[5].set_ylabel('y_ii - y_last')

    

#%% Plotting y_best_ii - y_best
f, ax = plt.subplots(3, 5, sharex=True, sharey=True)
ax = np.ravel(ax)


for ii in range(bo_args['nb_iter']):
    these_coords = bo_args['initial_design_numdata'] + ii*nn_mask
    these_coords = [int(cc) for cc in np.ravel(these_coords)]


    y_best_ii = [min(opt[ii].optimiser.Y[:these_coords[ii]]) for ii in range(shape[0]**2)]
    y_best_ii = np.squeeze(y_best_ii).reshape(*shape)
    diff = np.abs(y_best_ii - y_best)
    
    im = ax[ii].pcolor(diff, vmin=0.0,vmax=2)
    f.colorbar(im, ax=ax[ii])
    ax[ii].set_title('iter' + str(ii))
    
    if ii > 9:
        ax[ii].set_xlabel('x2-x1 (au)')
    if ii%5==0:
        ax[ii].set_ylabel('x3-x2 (au)')
ax[5].set_ylabel('best|ii - best_all')





#%% Plotting y_ii
f, ax = plt.subplots(3, 5, sharex=True, sharey=True)
ax = np.ravel(ax)


for ii in range(bo_args['nb_iter']):
    these_coords = bo_args['initial_design_numdata'] + ii*nn_mask
    these_coords = [int(cc) for cc in np.ravel(these_coords)]
    
    y_ii = [opt[ii].optimiser.Y[these_coords[ii]] for ii in range(shape[0]**2)]
    y_ii = np.squeeze(y_ii).reshape(*shape)
    
    
    im = ax[ii].pcolor(rescale(y_ii), vmin=0.0,vmax=2)
    f.colorbar(im, ax=ax[ii])
    ax[ii].set_title('iter' + str(ii))
    
    if ii > 9:
        ax[ii].set_xlabel('x2-x1 (au)')
    if ii%5==0:
        ax[ii].set_ylabel('x3-x2 (au)')
ax[5].set_ylabel('y_ii')



#%% Importing and plotting data
import joblib
fname = 'h3_full_circs_init{}_iter{}_method.dmp'.format(nb_init,nb_iter) + method[-2:]
with open('h3_chain_data.dmp', 'wb') as f:
    data = {'ed':ed_energies_mat,
            'opt':opt_energies_mat,
            'optims':runner.optim_list}
    joblib.dump(data, f)