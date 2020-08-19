import os
import time
import numpy as np
import qiskit as qk
import qcoptim as qc
import joblib as jbl
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qextras.chip import find_best_embedding_circuit_backend,embedding_to_initial_layout

pi = np.pi
# ------------------------------------------------------
# General Qiskit related helper functions
# ------------------------------------------------------
method = '2d' # 'independent_plus_random_4' or '2d'
backend = 4
nb_init = 1
nb_iter = 1
shape = (2,2)
positions = np.linspace(0.3, 1.8, shape[0])



fname = 'h3_circs_prmBool.dmp'
fname = 'H3_non_periodic_fewerprms.dmp'
if fname not in os.listdir():
    raise FileNotFoundError("This test assumes you have circ dmp file: 'h3_circs_prmBool.dmp'")
data = jbl.load(fname)
ansatz = qc.ansatz.AnsatzFromQasm(data[0]['qasm'], data[0]['should_prm'])



bem = qc.utilities.BackendManager()
bem.get_backend(backend, inplace=True)
provider = qk.IBMQ.get_provider(hub = 'ibmq', group='samsung', project='imperial')
ibm_backend = provider.get_backend(bem.LIST_OF_DEVICES[backend-1])

if backend != 4:
    embedding = find_best_embedding_circuit_backend(ansatz.circuit,
                                                    ibm_backend,
                                                    mode='111')
    initial_layout = embedding_to_initial_layout(embedding)
    mitigation = CompleteMeasFitter
    noise_model = None
else:
    initial_layout = None
    mitigation = CompleteMeasFitter
    noise_model = qc.utilities.gen_quick_noise()
inst = bem.gen_instance_from_current(initial_layout=initial_layout,
                                     nb_shots=2**9,
                                     optim_lvl=2,
                                     measurement_error_mitigation_cls=mitigation,
                                     noise_model=noise_model)
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
        cost_list.append(cst)
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

#%%
scf_energy = np.zeros(shape[0]**2);ii = 0
for dx1 in positions:
    for dx2 in positions:
        qasm_str = data[0]['qasm']
        bool_lst = data[0]['should_prm']
        coords = (dx1, dx2)
        atom = 'H 0 0 0; H 0 0 {}; H 0 0 {}'.format(*np.cumsum(coords))
        ansatz = qc.ansatz.AnsatzFromQasm(qasm_str, bool_lst)

        cst = qc.cost.ChemistryCost(atom, ansatz, inst, verbose=False)
        # cost_list.append(cst)
        scf_energy[ii] = cst._min_energy
        ii+=1
scf_energy = np.reshape(scf_energy, shape)
#%% Create cost functions
ansatz = qc.ansatz.AnsatzFromQasm(data[0]['qasm'], data[0]['should_prm'])
ansatz = qc.ansatz.RandomAnsatz(4,2)
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
results = inst.execute(runner.circs_to_exec,had_transpiled=True)
print('took {:2g} s to run inits'.format(time.time() - t))

t = time.time()
runner.init_optimisers(results)
print('took {:2g} s to init the {} optims from {} points'.format(time.time() - t, shape[0]**2, bo_args['initial_design_numdata']))

for ii in range(bo_args['nb_iter']):
    t = time.time()
    runner.next_evaluation_circuits()
    print('took {:2g} s to optim acq function'.format(time.time()  - t))

    t = time.time()
    results = inst.execute(runner.circs_to_exec,had_transpiled=True)
    print('took {:2g} s to run circs'.format(time.time()  - t))

    t = time.time()
    runner.update(results)
    print('took {:2g} s to run {}th update'.format(time.time() - t, ii))


x_opt_pred = [opt.best_x for opt in runner.optim_list]
runner.shot_noise(x_opt_pred, nb_trials=1)
results = inst.execute(runner.circs_to_exec,had_transpiled=True)
runner._last_results_obj = results
opt_energies = runner._results_from_last_x()
opt_energies_mat = np.reshape(np.squeeze(opt_energies), shape)


f , ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
im = ax[0].pcolor(rescale(scf_energy),
                  vmin = -1, vmax = 1.4)
ax[0].set_title('log(exact energy)')
ax[0].set_aspect('equal')
ax[0].set_ylabel('x3-x2 (A)')
ax[0].set_xlabel('x2-x1 (A)')
f.colorbar(im, ax=ax[0])


im = ax[1].pcolor(rescale(opt_energies_mat),
                  vmin = -1, vmax = 1.4)
ax[1].set_title('log(VQE energy)')
ax[1].set_aspect('equal')
ax[1].set_xlabel('x2-x1 (A)')
f.colorbar(im, ax=ax[1])



#%% Save data
# Importing and plotting data
import joblib
fname = 'h3_lin_circs_144_init{}_iter{}_method{}_large3.dmp'.format(nb_init,nb_iter,method[-2:])
with open(fname, 'wb') as f:
    data = {'shape':shape,
            'positions':positions,
            'opt':opt_energies_mat,
            'scf':scf_energy,
            'optims':runner.optim_list}
    joblib.dump(data, f)



# #%% Plotting y_ii - y_fin
# # Set up fig stuff
# f, ax = plt.subplots(3, 5, sharex=True, sharey=True)
# ax = np.ravel(ax)


# # Get optim list, and 'best' values of y (approx as last)
# opt = runner.optim_list
# y_last = np.squeeze([opt[ii].optimiser.Y[-1] for ii in range(shape[0]**2)]).reshape(*shape)
# y_best = [min(opt[ii].optimiser.Y) for ii in range(shape[0]**2)]
# y_best = np.reshape(y_best, shape)

# # Gen NN mask (different iter for each point on grid)
# if '2d' in method:
#     x = np.ones(shape)
#     x = np.pad(x, [1,1], mode='constant', constant_values=[0,0])
#     y = [[0, 1, 0], [1,1, 1], [0, 1, 0]]
#     nn_mask = convolve2d(x, y, 'valid')
# else:
#     nn_mask = 5*np.ones(shape)

# for ii in range(bo_args['nb_iter']):
#     these_coords = bo_args['initial_design_numdata'] + ii*nn_mask
#     these_coords = [int(cc) for cc in np.ravel(these_coords)]

#     y_ii = [opt[ii].optimiser.Y[these_coords[ii]] for ii in range(shape[0]**2)]
#     y_ii = np.squeeze(y_ii).reshape(*shape)

#     diff = np.abs(y_ii - y_last)

#     im = ax[ii].pcolor(diff, vmin=0.0,vmax=2)
#     f.colorbar(im, ax=ax[ii])
#     ax[ii].set_title('iter' + str(ii))

#     if ii > 9:
#         ax[ii].set_xlabel('x2-x1 (au)')
#     if ii%5==0:
#         ax[ii].set_ylabel('x3-x2 (au)')
# ax[5].set_ylabel('y_ii - y_last')



# #%% Plotting y_best_ii - y_best
# f, ax = plt.subplots(3, 5, sharex=True, sharey=True)
# ax = np.ravel(ax)


# for ii in range(bo_args['nb_iter']):
#     these_coords = bo_args['initial_design_numdata'] + ii*nn_mask
#     these_coords = [int(cc) for cc in np.ravel(these_coords)]


#     y_best_ii = [min(opt[ii].optimiser.Y[:these_coords[ii]]) for ii in range(shape[0]**2)]
#     y_best_ii = np.squeeze(y_best_ii).reshape(*shape)
#     diff = np.abs(y_best_ii - y_best)

#     im = ax[ii].pcolor(diff, vmin=0.0,vmax=2)
#     f.colorbar(im, ax=ax[ii])
#     ax[ii].set_title('iter' + str(ii))

#     if ii > 9:
#         ax[ii].set_xlabel('x2-x1 (au)')
#     if ii%5==0:
#         ax[ii].set_ylabel('x3-x2 (au)')
# ax[5].set_ylabel('best|ii - best_all')





# #%% Plotting y_ii
# f, ax = plt.subplots(3, 5, sharex=True, sharey=True)
# ax = np.ravel(ax)


# for ii in range(bo_args['nb_iter']):
#     these_coords = bo_args['initial_design_numdata'] + ii*nn_mask
#     these_coords = [int(cc) for cc in np.ravel(these_coords)]

#     y_ii = [opt[ii].optimiser.Y[these_coords[ii]] for ii in range(shape[0]**2)]
#     y_ii = np.squeeze(y_ii).reshape(*shape)


#     im = ax[ii].pcolor(rescale(y_ii), vmin=0.0,vmax=2)
#     f.colorbar(im, ax=ax[ii])
#     ax[ii].set_title('iter' + str(ii))

#     if ii > 9:
#         ax[ii].set_xlabel('x2-x1 (au)')
#     if ii%5==0:
#         ax[ii].set_ylabel('x3-x2 (au)')
# ax[5].set_ylabel('y_ii')



# #%% Importing and plotting data
# import joblib
# fname = 'h3_lin_circs_paris_init{}_iter{}_method{}.dmp'.format(nb_init,nb_iter,method[-2:])
# with open(fname, 'wb') as f:
#     data = {'ed':ed_energies_mat,
#             'opt':opt_energies_mat,
#             'optims':runner.optim_list}
#     joblib.dump(data, f)
