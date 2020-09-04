"""Created on Tue Feb 25 19:47:04 2020

@author: fred
#TODO: factorized the testing
#TODO: generate group elements
#TODO: Implement measurement of expectations
"""
import qutip as qt
import numpy as np
import itertools as it
import functools as ft
from qutip.qip.operations import cphase
from . import utilities as ut

__all__ = [
    'weights_and_settings'
]

def weights_and_settings(state, nb_qubits = 3, **args):
    """
    Returns: (settings, weights) pair, where settings is a list of Pauli 
    string measure settings, and weights is a list of corresponding weights to 
    evaluate the projector
    
    Parameters:
    -------
    state : string
        'ghz', 'cluster' supported for now
    nb_qubits : int
        number of qubits in the state
    **args can be passed into state generator
    """
    if state == 'graph':
        graph_test = gen_graph_state(N=nb_qubits, **args)
        return gen_decomposition_paulibasis(graph_test, nb_qubits, threshold=1e-6, symbolic=True)
    elif state == 'ghz':
        state = gen_ghz(nb_qubits)
        return gen_decomposition_paulibasis(state, nb_qubits, threshold=1e-6, symbolic=True)    
      

# Define one-qubit state operators
zero = qt.qubits.qubit_states(1,[0])
one = qt.qubits.qubit_states(1,[1])
plus = (one + zero)/np.sqrt(2)
minus = (one - zero)/np.sqrt(2)
I, X, Y, Z = qt.identity([2]), qt.sigmax(), qt.sigmay(), qt.sigmaz()
Rx, Ry, Rz = qt.rx, qt.ry, qt.rz

#======================#
# Basis and decomposition
# TODO: more factorization - gen_tensored_basis, symbolic as an input, gen_decompo
# 
#======================#
# Computational basis
def gen_computbasis_proj(N=3):
    """ generate a list of the 2^N elements of computational basis proj"""
    op_comput = [zero * zero.dag(), one * one.dag()]
    return [tensor_listop(parts) for parts in it.product(*[op_comput]*N)]

def gen_computbasis_kets(N=3):
    """ generate a list of the 2^N elements of computational basis proj"""
    op_comput = [zero , one]
    return [tensor_listop(parts) for parts in it.product(*[op_comput]*N)]

def gen_computbasis(N=3):
    """ generate a list of the 4^N elements of computational basis"""
    op_comput = [zero * zero.dag(), zero * one.dag(), one * zero.dag(), one * one.dag() ]
    return [tensor_listop(parts) for parts in it.product(*[op_comput]*N)]

def gen_computbasis_symbolic(N=3):
    """ generate a list of the 4^N elements of computational basis symbols"""
    op_comput = [('0','0'), ('0','1'), ('1','0'), ('1','1')]
    return [str_to_proj(parts) for parts in it.product(*[op_comput]*N)]

def str_to_proj(string):
    """[('0','1'), ('1','1'), ('0','1')] -> '|010><111|' """
    final = ft.reduce(lambda x, y : [x[0]+ y[0], x[1]+ y[1]], string, ['',''])
    return '|' + final[0] + '><' + final[1] + '|'
    

# Pauli basis 
def gen_paulibasis(N=3):
    """ generate a list of the 4^N elements of the tensored Pauli basis"""
    op_pauli = [I, X, Y, Z]
    return [tensor_listop(parts) for parts in it.product(*[op_pauli]*N)]

def gen_paulibasis_symbolic(N=3):
    """ generate a list of the 4^N elements of the tensored Pauli basis in their 
    symbolic form, e.g. ['II', 'IX', ..., 'ZZ']"""
    op_pauli = ['1', 'x', 'y', 'z']
    return ["".join(parts) for parts in it.product(*[op_pauli]*N)]


#decompo / recompo
def gen_decomposition_paulibasis(state, N, threshold=0, symbolic=False):
    """ Find the decomposition  of a given state as \sum alpha_i P_i where P_i 
    are elements of the tensored Pauli basis and alpha_i = <state, P_i> / d
    with <A,B> the Hilbert Schmidt product 
    Return: (c, b) a list of components and basis operators associated
           if threshold > 0, only return c[i], b[i] such that np.abs(c[i]) >= threshold
           if symbolic = True, b are given as symbolic string 
    """
    basis = gen_paulibasis(N)
    norm_basis = np.power(2,N)
    comp = assert_and_recast_to_real([qt.expect(op, state)/norm_basis for op in basis])
    if symbolic: 
        basis = gen_paulibasis_symbolic(N)
    if threshold > 0:
        above_threshold = np.abs(comp) >= threshold 
        comp = [c for c, t in zip(comp, above_threshold) if t]
        basis = [b for b, t in zip(basis, above_threshold) if t]
    return (comp, basis)

def gen_recomposition_paulibasis(components, N):
    """ Recompose an operator given its components in a Pauli basis
    """
    basis = gen_paulibasis(N)
    return gen_op_from_decompo(components, basis)

def gen_op_from_decompo(components, basis):
    """ Resconstruct an operator from its components in a given basis"""
    return weighted_sum_listop(components, basis)

#======================#
# General functions for stabilizer
#======================#
# Witnesses and cost functions
def gen_w1(N, stab_gen_func, **args):
    """ generate a witness for stabilizer based on 2 measurement settings:
    w1 = 3 I - 2[(G1+I)/2 + Prod_{k>1} (Gk + I)/2]
    where Gk is the k-th element of the stabilizer generators set
    + stab_gen_func: functions generating the set of stabilizer generators 
          elements. Signature: (N, **args) >> list(qtip.operators)
    + 
    """
    gen = stab_gen_func(N, **args)
    id_2n = qt.identity([2]*N)
    w1 = 3 * id_2n - (gen[0] + id_2n)
    w1 += prod_listop([0.5*(g + id_2n) for g in gen[1:]])    
    return w1

def gen_w2(N, stab_gen_func, **args):
    """ generate a witness for stabilizer based on 2 measurement settings:
    W2 = (N-1) I - sum G_k.
    Same input/output as gen_w1
    """
    gen = stab_gen_func(N, **args)
    w2 = (N-1) * qt.identity([2]*N) - sum_listop(gen)
    return w2

def gen_F1(N, stab_gen_func, **args):
    """ generate a cost function operator based on witness 1, this function 
    is maximized only when the target state is reached, o.w. strictly smaller 
    than the fidelity
    """
    gen = stab_gen_func(N, **args)
    id_2n = qt.identity([2]*N)
    f1 = 0.5 * (gen[0] + id_2n) + prod_listop([0.5*(g + id_2n) for g in gen[1:]])    
    f1 -= id_2n
    return f1

def gen_F2(N, stab_gen_func, **args):
    """ Same as gen_F1 but based on w2 """
    gen = stab_gen_func(N, **args)
    f2 = sum_listop(gen) - (N-1) * qt.identity([2]*N)
    return f2

#======================#
# GHZ
#======================#
# generation
def gen_ghz(N=3, angle=0):
    """ Generate a GHZ state of the form |00..00> + exp(i * angle) |11..11> """
    n_zero = tensor_listop(([zero]*N))
    n_one = tensor_listop(([one]*N))
    return 1/np.sqrt(2) * (n_zero+ np.exp(1.0j*angle) * n_one)

def gen_proj_ghz(N=3):
    """ gen the projector on the GHZ state"""
    ghz = gen_ghz(N)
    return ghz * ghz.dag()

def gen_random_state(N=3):
    """ retuns a random (normalised) state in the Hilbertspace - Random normal phases, gaussian distributed amplitues"""
    computbasis_kets = gen_computbasis_kets(N);
    amplitudes = np.random.normal(0, 1, 2**N)
    phases = 2*np.pi * np.random.rand(2**N)
    random_state = sum([aa*np.exp(1j * pp)*ss for aa, pp, ss in zip(amplitudes, phases, computbasis_kets)])
    return random_state / (random_state.norm())

#Stabilizer generators, and group elements
def gen_stab_gen_ghz(N=3, symbolic=False):
    """ Generate one set of stabilizer GENERATORS for the GHZ state
    <X1...XN, Z1Z2,...,ZN-1ZN> 
        if symbolic = False(True) each element is a qutip object(a symbolic string)
    """
    if symbolic: 
        X1, Z1, I1 = 'X', 'Z', 'I' #i.e. do not use qutip operators but symbols
        func_accumulate = lambda x: "".join(x)
    else:
        X1, Z1, I1 = X, Z, I
        func_accumulate = tensor_listop
    list_gen = []
    list_gen.append(func_accumulate([X1]*N))
    for i in range(N-1):
        gen_tmp = []
        if i >0: gen_tmp += [I1]*i
        gen_tmp += [Z1, Z1]
        if i<N-1: gen_tmp += [I1]*(N-i-2)
        list_gen.append(func_accumulate(gen_tmp))
    return list_gen

def gen_stab_group_ghz(N=3, symbolic=False):
    """ Generate 2^N elements of the stabilizer GROUP for the GHZ state
    TODO: 
    """
    pass

# Witnesses and cost functions
def gen_w1_ghz(N=3):
    """ w1 for GHZ states"""
    return gen_w1(N, gen_stab_gen_ghz)

def gen_w2_ghz(N=3):
    """ w1 for GHZ states"""
    return gen_w2(N, gen_stab_gen_ghz)

def gen_F1_ghz(N=3):
    """ F1 for ghz states"""
    return gen_F1(N, gen_stab_gen_ghz)

def gen_F2_ghz(N=3):
    """ F2 for ghz states"""
    return gen_F2(N, gen_stab_gen_ghz)


#======================#
# Graph state
#======================#
#gen of the state
def gen_graph_state(N, edges = None):
    """ Generate a graph state corresponding to a given number of qubits N and
    list of edges: first(second) bit corresponds to the control(target). If list
    of edges is None, cyclic graph is assumed"""
    if edges == None:
        edges = ut.gen_cyclic_graph(N)
    init = qt.tensor([plus]*N)
    evol = prod_listop([cphase(np.pi, N, e[0], e[1]) for e in edges])
    return evol * init

def gen_proj_graph_state(N, edges):
    """ Generate a projector onto the graph state"""
    graph = gen_graph_state(N, edges)
    return graph * graph.dag()

#Stabilizer generators, and group elements
def gen_stab_gen_graph(N, edges, symbolic=False):
    """ Generate one set of stabilizer GENERATORS for a graph state specified 
    by a number of qubits and list of edges
    <K_1,..., K_N> with K_j = X_j \prod{i ~e(j)} Z_i  (i~j means i connected to j)
        
    + if symbolic = False(True) each element is a qutip object(a symbolic string)
    """
    if symbolic: 
        X1, Z1, I1 = 'X', 'Z', 'I' #i.e. do not use qutip operators but symbols
        func_accumulate = lambda x: "".join(x)
    else:
        X1, Z1, I1 = X, Z, I
        func_accumulate = tensor_listop
    list_op1 = [I1, Z1, X1]
    list_gen = []
    #list_gen.append(func_accumulate([X1]*N))
    for i in range(N):
        conn_tmp = connected(i, N, edges)
        conn_tmp[i] = 2
        gen_tmp = [list_op1[c] for c in conn_tmp]
        list_gen.append(func_accumulate(gen_tmp))
    return list_gen

# Witnesses and cost functions
def gen_w1_graph(N, edges):
    """ w1 for graph states"""
    return gen_w1(N, gen_stab_gen_graph,edges=edges)

def gen_w2_graph(N, edges):
    """ w1 for graph states"""
    return gen_w2(N, gen_stab_gen_graph,edges=edges)


def gen_F1_graph(N, edges):
    """ F1 for graph states"""
    return gen_F1(N, gen_stab_gen_graph, edges=edges)

def gen_F2_graph(N, edges):
    """ F2 for graph states"""
    return gen_F2(N, gen_stab_gen_graph,edges=edges)


def connected(j, N, edges):
    """ for the jth qubit out of N, return a length-N list connect with connect[i]
    = 0 if j is not connected to i or 1 if connected """
    list_connect = [e[1] if e[0]==j else e[0] for e in edges if (e[0]==j or e[1]==j)]
    connect = [1 if i in list_connect else 0 for i in range(N)]
    return connect


#======================#
# Expectations True and estimated
# TODO: add measurement capabilities / re-factor 
# 
#======================#
def estimate_op_proj(list_proj, list_states, nb_meas):
    """ Estimates values of projector operators PP = P
    >> return estimated values PxSxR array of estimated frequencies 
       with P the number of projectors, S the number of states
    TODO: can nb_repeat be incorporated
    TODO: can list_n_meas be incorporated
    """
    probas = np.array([[qt.expect(P,s) for s in list_states] for P in list_proj])
    freq = np.random.binomial(nb_meas, probas) /nb_meas
    estim = 2*freq
    return np.squeeze(estim)    
    
def estimate_op_bin(list_bin, list_states, nb_meas):
    """ Estimates values of binary operators O = P+ - P- with P+(-) projectors
    """
    exp_val = np.array([[qt.expect(P,s) for s in list_states] for P in list_bin])
    probas = (exp_val +1)/2
    freq = np.random.binomial(nb_meas, probas) /nb_meas
    estim = 2*freq-1
    return np.squeeze(estim)    






#======================#
# Utilities
#======================#
def sum_listop(list_op):
    """ (cumulative) sum over an arbitrary list of qtip operators
    [A,B,C] -> A+B+C 
    """
    return ft.reduce(lambda x,y : x+y, list_op, 0)

def weighted_sum_listop(list_op, list_weights):
    """ (cumulative) weighted sum over an arbitrary list of qtip 
    operators and weights
    [A,B,C], [1,2,3] -> A+2*B+3*C 
    """
    assert len(list_op) == len(list_weights)
    weighted_obs = [w*op for op, w in zip(list_op, list_weights)]
    return sum_listop(weighted_obs)

def prod_listop(list_op):
    """ (cumulative) product over an arbitrary list of qtip operators
    [A,B,C] -> AxBxC with x here the matrix mult
    """
    return ft.reduce(lambda x,y : x*y, list_op, 1)

def tensor_listop(list_op):
    """ (cumulative) tensor over an arbitrary list of qtip operators
    [A,B,C] -> AxBxC with x here the tensor product
    Comment: added to complete prod_listop/sum_listop
    """
    return qt.tensor(*list_op)

   
# Extra-stuff
def assert_and_recast_to_real(array):
    """ assert if imaginary part is negligeable and return real part"""
    assert np.allclose(np.imag(array), 0), "Assertion failed: non real array"
    return np.real(array)




#%% Test section
if __name__ == '__main__':
    # Decomposition of a graph state
    N_test = 6
    edges_test = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,0]] 
    graph_test = gen_graph_state(N=N_test, edges=edges_test)
    pauli_gs6 = gen_decomposition_paulibasis(graph_test, N_test, threshold=1e-6, symbolic=True)

    # Decomposition of a ghz state
    N_test = 2
    graph_test = gen_ghz(N_test)
    pauli_ghz2 = gen_decomposition_paulibasis(graph_test, N_test, threshold=1e-6, symbolic=True)    
    
    #TODO: re-factorize testing move somewhere else
    #-----#
    # Pauli decomposition
    #-----#
    N_test = 3
    ghz3 = gen_ghz(N_test)
    decomp_symb = gen_decomposition_paulibasis(ghz3, N_test, threshold=1e-6,symbolic=True)
    decomp_full = gen_decomposition_paulibasis(ghz3, N_test)
    recomp_full = gen_recomposition_paulibasis(decomp_full[0], N_test)
    assert recomp_full == ghz3 * ghz3.dag(), "Failed assertion 1: decomp/recomp in Pauli basis"

    #-----#
    # GHZ
    #-----#    
    #testing witnesses and cost functions for GHZ states (could be factorized 
    # as much as gen_w1 works for all stabilizer states)
    f1 = gen_F1_ghz(N_test)
    f2 = gen_F2_ghz(N_test)
    f = gen_proj_ghz(N_test)
    ev1 = assert_and_recast_to_real(np.linalg.eigvals((f1-f).full()))
    ev2 = assert_and_recast_to_real(np.linalg.eigvals((f2-f).full()))
    assert np.all(ev1 <= 1e-6), "fom1 should be <= f"
    assert np.all(ev2 <= 1e-6), "fom2 should be <= f"
    assert np.allclose(np.max(ev1),0), "there should be at least one 0 e.v. (corresponding to the target state)"
    assert np.allclose(np.max(ev2),0), "there should be at least one 0 e.v. (corresponding to the target state)"
    # add test on expect values
    
    #-----#
    # Graph state 4-qubit
    #-----#
    N_test = 4
    edges_test = [[0,1], [1,2], [2,3]] 
    graph_test = gen_graph_state(N=N_test, edges=edges_test)
    list_gen_test = gen_stab_gen_graph(N_test, edges_test, symbolic=False)
    assert np.allclose([(gen * graph_test - graph_test).norm() for gen in list_gen_test], 0), "one of the element is not a stabilizer"
    gen_decomposition_paulibasis(graph_test, N_test, threshold=1e-6, symbolic=True)
    #testing witnesses and cost functions for GHZ states
    f1 = gen_F1_graph(N_test, edges_test)
    f2 = gen_F2_graph(N_test, edges_test)
    f = gen_proj_graph_state(N_test, edges_test)
    ev1 = assert_and_recast_to_real(np.linalg.eigvals((f1-f).full()))
    ev2 = assert_and_recast_to_real(np.linalg.eigvals((f2-f).full()))
    assert np.all(ev1 <= 1e-6), "fom1 should be <= f"
    assert np.all(ev2 <= 1e-6), "fom2 should be <= f"
    assert np.allclose(np.max(ev1),0), "there should be at least one 0 e.v. (corresponding to the target state)"
    assert np.allclose(np.max(ev2),0), "there should be at least one 0 e.v. (corresponding to the target state)"
    
    
    #-----#
    # Graph state 4-qubit
    #-----#
    N_test = 6
    edges_test = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,0]] #2-D graph state
    graph_test = gen_graph_state(N=N_test, edges=edges_test)
    #gen_stab_gen_graph(N_test, edges_test, symbolic=True)
    list_stab = gen_stab_gen_graph(N_test, edges_test, symbolic=False)
    # seems right
    gen_decomposition_paulibasis(graph_test, N_test, threshold=1e-6, symbolic=True)
    #testing witnesses and cost functions for GHZ states
    f1 = gen_F1_graph(N_test, edges_test)
    f2 = gen_F2_graph(N_test, edges_test)
    f = gen_proj_graph_state(N_test, edges_test)
    ev1 = assert_and_recast_to_real(np.linalg.eigvals((f1-f).full()))
    ev2 = assert_and_recast_to_real(np.linalg.eigvals((f2-f).full()))
    assert np.all(ev1 <= 1e-6), "fom1 should be <= f"
    assert np.all(ev2 <= 1e-6), "fom2 should be <= f"
    assert np.allclose(np.max(ev1),0), "there should be at least one 0 e.v. (corresponding to the target state)"
    assert np.allclose(np.max(ev2),0), "there should be at least one 0 e.v. (corresponding to the target state)"
    
    # 
    nb_q = 6
    x_rdm = np.array([1.70386471,1.38266762,3.4257722,5.78064,3.84102323,2.37653078])
    init = qt.tensor(*[zero]*nb_q)
    rot_init = qt.tensor(*[qt.ry(N=1, phi=x) for n,x in enumerate(x_rdm)])
    cz = [cphase(np.pi, nb_q, c, t) for c, t in edges_test]
    entangl = prod_listop(cz)
    fin = entangl * rot_init * init
    qt.expect(f, fin)
    qt.expect(f1, fin)
    qt.expect(f2, fin)
    
    exp_stab = [qt.expect(s, fin) for s in list_stab]
    
    N_test = 6
    edges_test = [[i, i+1]for i in range(N_test-1)] + [[N_test-1, 0]] 
    graph_test = gen_graph_state(N=N_test, edges=edges_test)
    #gen_decomposition_paulibasis(graph_test, N = N_test, threshold=1e-6, symbolic=True)
    
