#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:41:03 2020

@author: fred
Goal: compare different fom for GHZ and graph states in terms of 
 + bias/variance (produce fig)
 + concentration (produce stat + scaling with N)
 
TODO:
    + generations of random state get more close to the target
    + implement sampling/measurements strategies
    + look at statistical properties

"""
import utilities_stabilizer as ut
import qutip as qt
import numpy as np
import matplotlib.pylab as plt
import numpy as np
from scipy.stats import unitary_group


#####  Part I  #####
# Look at GHZ states
####################
N_ghz=8
N_states = 200

ghz = ut.gen_ghz(N_ghz)
decomp = ut.gen_decomposition_paulibasis(ghz, N_ghz, threshold=1e-6, symbolic=True)
dims_ghz = ghz.dims
#states_haar = [qt.rand_ket_haar(N=2**N_ghz, dims=dims_ghz) for _ in range(N_states)]


close_states = []
for ii in range(N_states):
    H = qt.rand_herm(2**N_ghz, dims = [[2]*N_ghz, [2]*N_ghz])
    t = 0.15*np.random.rand()
    U = (-1j*H*t).expm()
    close_states.append(U*ghz)
    print(ii)
states_haar = close_states


F = ut.gen_proj_ghz(N_ghz)
F1 = ut.gen_F1_ghz(N_ghz)
F2 = ut.gen_F2_ghz(N_ghz)

list_f = np.array([qt.expect(F, st) for st in states_haar])
list_f1 = np.array([qt.expect(F1, st) for st in states_haar])
list_f2 = np.array([qt.expect(F2, st) for st in states_haar])


fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(list_f, list_f1, label='w1')
ax.scatter(list_f, list_f2, label='w2')
ax.plot(list_f, list_f, 'r', label='fid')
ax.legend()
ax.set_title('witnesses - {} qubits'.format(N_ghz))
ax.set_xlabel('Fid')
ax.set_ylabel('Witness value')
ax.axis([0, 1, -2, 1])
#%%
#####  Part Ia  #####
# Look at the concentration
#####################
plt.hist(list_f)
plt.hist(list_f1)
plt.hist(list_f2)

avg, std = np.average(list_f), np.std(list_f)
avg1, std1 = np.average(list_f1), np.std(list_f1)
avg2, std2 = np.average(list_f2), np.std(list_f2)
std_norm = std / (1-avg)
std_norm1 = std1 / (1-avg1)
std_norm2 = std2 / (1-avg2)
print(std_norm)
print(std_norm1)
print(std_norm2)

#3,4,5,6
res = np.array([[0.258276107091469, 0.3161054803737825, 0.3161054803737825],
                [0.1257447436332672,0.17622056376478779,0.19169664598402428],
                [0.06347733159484138, 0.10669941905274924, 0.12274957816578666],
                [0.03141659463343181, 0.06703660442251375, 0.07872189019318448],
                [0.015467538548765258, 0.04435145866191629,0.050931430704355414],
                [0.007794890212225759,0.03020906228369326,0.03331424973112738],
                [0.0038866049186859074,0.021085311335449268,0.0222141975578163],
                [0.0019566275973149382, 0.014968986939285224, 0.014815183106616713],
                [0.0009681666870207654,0.010561279603527155,0.009775006277963981]
])
nb_q = [1,2,3,4,5,6,7,8,9]

log_res = np.log(res)
log_q = np.log(nb_q)

fig, ax = plt.subplots()
ax.plot(nb_q, log_res[:,0])
ax.plot(nb_q, log_res[:,1])
ax.plot(nb_q, log_res[:,2])


fig, ax = plt.subplots()
ax.plot(log_q, log_res[:,0],'o--')
ax.plot(log_q, log_res[:,1],'s--')
ax.plot(log_q, log_res[:,2],'v--')


fig, ax = plt.subplots()
ax.plot(nb_q[1:], res[1:,0]/res[:-1,0],'o--')
ax.plot(nb_q[1:], res[1:,1]/res[:-1,1],'s--')
ax.plot(nb_q[1:], res[1:,2]/res[:-1,2],'v--')
ax.set_yscale("log")

#####  Part Ib  #####
# Look at statistical properties 
#####################
stabgen_op = ut.gen_stab_gen_ghz(N_ghz)
N_repeat = 100
N_meas = 50

# look at stat properties of operators themselves for 
# They have the same properties
estimates = [ut.estimate_op_bin(stabgen_op, states_haar, N_meas) for _ in range(N_repeat)]
#stats_avg = np.mean(estimates, axis=0)
#stats_std = np.std(estimates, axis=0)
#plt.hist(stats_std[0])
#plt.hist(stats_std[1])
#plt.hist(stats_std[2])
#np.average(stats_std[1])
#plt.hist(stats_avg[0])
#plt.hist(stats_avg[1])
#plt.hist(stats_avg[2])

expected = np.array([[qt.expect(op, st) for st in states_haar]for op in stabgen_op])
stats_avg = np.mean(expected, axis=1)
stats_std = np.std(expected, axis=1)

## strat estimate 1:



#%% Witness as shot noise 

import qcoptim as qc

try:
    bem
except:
    bem = qc.utilities.BackendManager();

    
inst = bem.gen_instance_from_current(1, 0)
inst_exact = bem.gen_instance_from_current(2**13, 0)

creator = qc.ansatz._SandwitchAnsatzes()
ghz_circ = creator.GHZ_plus_rotation(nb_qubits=6, 
                                     init_rotation='u3', 
                                     final_rotation='u3')



x_sol = np.zeros((1,ghz_circ.nb_params))
cost_full = qc.cost.StateFidelityCost('ghz', ghz_circ, inst = inst)
cost_w1 = qc.cost.GHZWitness1Cost(ghz_circ, inst = inst)
cost_w2 = qc.cost.GHZWitness2Cost(ghz_circ, inst = inst)


actual_0 = qc.cost.StateFidelityCost('ghz', ghz_circ, inst = inst_exact)
actual_1 = qc.cost.GHZWitness1Cost(ghz_circ, inst = inst_exact)
actual_2 = qc.cost.GHZWitness2Cost(ghz_circ, inst = inst_exact)



total_budget = 130
total_states = 100
data = []
char_f = []
for ii in range(total_states):
    print(ii)
    x_param = .1*np.random.rand(ghz_circ.nb_params)
    
   # est_0 = np.mean(cost_full([x_param]*(total_budget//9)))
    est0 = 0
    est_1 = np.mean(cost_w1([x_param]*(total_budget//2)))
    est_2 = np.mean(cost_w2([x_param]*(total_budget//2)))
        
    target_0 = np.mean(actual_0(x_param))
    target_1 = np.mean(actual_1(x_param))
    target_2 = np.mean(actual_2(x_param))
    
    dev0 = target_0 - est_0
    dev1 = target_1 - est_1
    dev2 = target_2 - est_2
    
    data.append([dev0, dev1, dev2])
    char_f.append(target_0)

data = np.array(data)
    
#%% Plotting
plt.plot(data[:,0], 'r', label='fid')
plt.plot(data[:,1], 'b', label='w1')
plt.plot(data[:,2], 'g', label='w2')
plt.xlabel('100 random states')
plt.title('Deviation from sampeled witness to full witnes')
plt.ylabel('estimator - mean')


plt.legend()

fig, ax = plt.subplots(1, 3)
ax[0].hist(data[:,0]);ax[0].set_title('fid')
ax[1].hist(data[:,1]);ax[1].set_title('w1')
ax[2].hist(data[:,2]);ax[2].set_title('w2')
ax[0].axis([-0.025, 0.025, 0, 20])
ax[1].axis([-0.025, 0.025, 0, 20])
ax[2].axis([-0.025, 0.025, 0, 20])