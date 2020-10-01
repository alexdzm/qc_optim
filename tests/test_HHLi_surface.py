import os
import time
import joblib
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
method = '2d1.5' # 'independent_plus_random_4' or '2d'
backend = 5
nb_init = 30
nb_iter = 30
shape = (8, 8)
positionsHH = np.linspace(0.2, 2.5, shape[0])
positionsLiH = np.linspace(0.4, 4, shape[0])



fname = 'HHLi_circs_18prms.dmp'
if fname not in os.listdir():
    raise FileNotFoundError("This test assumes you have circ dmp file: 'h3_circs_prmBool.dmp'")
data = jbl.load(fname)
ansatz = qc.ansatz.AnsatzFromQasm(data[0]['qasm'], data[0]['should_prm'])



bem = qc.utilities.BackendManager()
bem.get_backend(backend, inplace=True)
provider = qk.IBMQ.get_provider(hub = 'ibmq', group='samsung', project='imperial')
ibm_backend = provider.get_backend(bem.LIST_OF_DEVICES[backend-1])

if backend != 5:
    embedding = find_best_embedding_circuit_backend(ansatz.circuit,
                                                    ibm_backend,
                                                    mode='010')
    initial_layout = embedding_to_initial_layout(embedding)
    mitigation = CompleteMeasFitter
    noise_model = None
else:
    initial_layout = None
    mitigation = None # CompleteMeasFitter
    noise_model = None # qc.utilities.gen_quick_noise()
inst = bem.gen_instance_from_current(initial_layout=initial_layout,
                                     nb_shots=2**9,
                                     optim_lvl=2,
                                     measurement_error_mitigation_cls=mitigation,
                                     noise_model=noise_model)
rescale = lambda x: (np.log(x +9))

if True:
    scf_energy = np.zeros(np.product(shape))
    cir_energy = np.zeros(np.product(shape))
    cost_list, wpo_list = [], []
    for ii, cc in enumerate(data):
        qasm_str = cc['qasm']
        bool_lst = cc['should_prm']
        coords = cc['physical_prms']
        coords = [abs(c) for c in coords]
        wpo_list.append(qc.utilities.get_H_chain_qubit_op(coords))


        atom = 'H 0 0 0; H 0 0 {}; Li 0 0 {}'.format(*np.cumsum(coords))
        ansatz = qc.ansatz.AnsatzFromQasm(qasm_str, bool_lst)

        cst = qc.cost.ChemistryCost(atom, ansatz, inst, verbose=False)
        cost_list.append(cst)
        scf_energy[ii] = cst._min_energy
        cir_energy[ii] = np.squeeze(cst(ansatz._x_sol))

        print('Min energy from pySCF   : ' + str(scf_energy[ii]))
        print('Min energy from our circ: ' + str(cir_energy[ii]))
        print('-----------')

    plt_scf = np.reshape(scf_energy, shape)
    plt_cir = np.reshape(cir_energy, shape)
    f , ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    im = ax[0].pcolor(rescale(plt_scf)) 
    ax[0].set_title('scf energies (log scale)')
    ax[0].set_aspect('equal')
    f.colorbar(im, ax=ax[0])


    im = ax[1].pcolor(rescale(plt_cir))
    ax[1].set_title('circuit energies (log scale)')
    ax[1].set_aspect('equal')   
    f.colorbar(im, ax=ax[1])



wpo_list = [qc.utilities.get_HHLi_qubit_op(*[dx1,dx2]) for dx1 in positionsHH for dx2 in positionsLiH]
wpo_list = qc.utilities.enforce_qubit_op_consistency(wpo_list)



#%% Create cost functions
ansatz = qc.ansatz.AnsatzFromQasm(data[0]['qasm'], data[0]['should_prm'])
# ansatz = qc.ansatz.RandomAnsatz(4,2)
# print('warning - this is a random ansatz')
cost_list = [qc.cost.CostWPO(ansatz, inst, ww) for ww in wpo_list]
ed_energies_mat = [c._min_energy for c in cost_list]
ed_energies_mat = np.reshape(ed_energies_mat, shape)

domain = np.array([(0, 2*pi) for i in range(cost_list[0].ansatz.nb_params)])
bo_args = qc.utilities.gen_default_argsbo(f=lambda: 0.5,
                                          domain=domain,
                                          nb_init=nb_init,
                                          eval_init=False)
bo_args['nb_iter'] = nb_iter
bo_args['acquisition_weight'] = 5
bo_args['acquisition_weight_lindec'] = False


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
    if ii > nb_iter * .8:
        bo_args['acquisition_weight'] = .2

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
im = ax[0].pcolor((plt_scf),
                  vmin = -8.6, vmax = -7)
ax[0].set_title('log(exact energy)')
ax[0].set_aspect('equal')
ax[0].set_ylabel('x3-x2 (A)')
ax[0].set_xlabel('x2-x1 (A)')
f.colorbar(im, ax=ax[0])


im = ax[1].pcolor((opt_energies_mat),
                  vmin = -8.6, vmax = -7)
ax[1].set_title('log(VQE energy)')
ax[1].set_aspect('equal')
ax[1].set_xlabel('x2-x1 (A)')
f.colorbar(im, ax=ax[1])


#%% Save data
# Importing and plotting data
fname = 'HHLi_sim_64_init{}_iter{}_method{}_optAnsatz.dmp'.format(nb_init,nb_iter,method[-2:])
with open(fname, 'wb') as f:
    data = {'shape':shape,
            'positions':[positionsHH, positionsLiH],
            'opt':opt_energies_mat,
            'scf':plt_scf,
            'circ':plt_cir}
    joblib.dump(data, f)


#%% 

fname = 'HHLi_sim_64_init15_iter10_method.5_optAnsatz.dmp'
fname = 'HHLi_sim_64_init20_iter20_method.5_optAnsatz.dmp'


#%% plotting results
data = joblib.load(fname)
scf_energy = data['scf']
opt_energies_mat = data['opt']
positions = data['positions']
positionsHH = positions[0]
positionsLiH = positions[1]

rescale = lambda x: np.log(x +9)


# Plot line by line
f , ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
im = ax[0].plot(positionsLiH, (scf_energy))
ax[0].set_title('(exact energy)')
ax[0].set_xlabel('x2-x1 (A)')
ax[0].legend(['HH: ' + str(round(p, 2)) for p in positionsHH], loc='upper right')


im = ax[1].plot(positionsLiH, opt_energies_mat)
ax[1].set_title('(VQE energy)')
ax[1].set_xlabel('x2-x1 (A)')
f.legend()

# Plot surface
f , ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
im = ax[0].pcolor((scf_energy), vmin=-9, vmax=-7)
ax[0].set_title('SCF energy')
ax[0].set_aspect('equal')
ax[0].set_ylabel('x3-x2 (A)')
ax[0].set_xlabel('x2-x1 (A)')
f.colorbar(im, ax=ax[0])
     

im = ax[1].pcolor(opt_energies_mat , vmin = -9, vmax = -7)
ax[1].set_title('VQE energy: Paris')
ax[1].set_aspect('equal')
ax[1].set_xlabel('x2-x1 (A)')
f.colorbar(im, ax=ax[1])


# Plot errors
f , ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

ax[0].plot(positionsLiH, opt_energies_mat - scf_energy)
ax[0].set_xlabel('x2-x1')
ax[0].set_ylabel('Eopt - Eexact')
ax[0].set_title('actual VQE error')
f.legend(['d32:'+str(round(p, 2)) for p in positionsHH])
f.legend()

ax[1].plot(positionsLiH, opt_energies_mat - plt_cir)
ax[1].set_xlabel('x2-x1')
ax[1].set_ylabel('Eopt - Ecirc')
ax[1].set_title('actual VQE error')
f.legend(['d32:'+str(round(p, 2)) for p in positionsHH])
f.legend()





