import os
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
cost_list = []
for ii, cc in enumerate(data):
    qasm_str = cc['qasm']
    bool_lst = cc['should_prm']
    coords = cc['prm']

    
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


# domain = np.array([(0, 2*pi) for i in range(cost_list[0].nb_params)])
# bo_args = qc.utilities.gen_default_argsbo(f=cst,
#                                           domain=domain,
#                                           nb_init=15,
#                                           eval_init=False)
# bo_args['nb_iter'] = 15 


# runner = qc.optimisers.ParallelRunner(cost_list, 
#                                       qc.optimisers.MethodBO,
#                                       optimizer_args = bo_args, 
#                                       share_init = True, 
#                                       method = '2d')