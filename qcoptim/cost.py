#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:11:28 2020

@author: fred

TODO: (MAYBENOT) Move bind_parameters to utils as it's used in optimisers
TODO: (SOON) Reimplement the max_circuit rule
TODO: (SOON) implement more general graph states 
TODO: (SOON) PROBLEM WITH WitnessesCost1 SHOULD NOT BE USED
TODO: (LATER) ability to deal with different number of shots 
TODO: (LATER) implement sampling (of the measurement settings) strategy



CHANGES
* Cost now conforms to CostInterface
* Cost.meas_func now changed to cost.evaluate_cost
* cost.evaluate_cost is now a general function that calls self._meas_func that was 
    generated in the sub classes
    cost.evaluate_cost accepts a qk.results OBJECT IFF
* Ansatz inputs now it's own class that holds lots of useful info

Choice of noise_models, initial_layouts, nb_shots, etc.. is done through the 
quantum instance passed when initializing a Cost, i.e. it is outside of the
scope of the classes here
"""

# list of * contents
__all__ = [
    'CostInterface',
    'GenericCost',
    'Cost',
    'compare_layout',
    'OneQProjZ',
    'OneQXYZ',
    'GHZPauliCost',
    'GHZWitness1Cost',
    'GHZWitness2Cost',
    'GraphCyclPauliCost',
    'GraphCyclWitness1Cost',
    'GraphCyclWitness2Cost',
    'GraphCyclWitness2FullCost',
    'GraphCyclWitness3Cost',
    'freq_even',
    'expected_parity',
    'get_substring',
    'append_measurements',
    'gen_meas_circuits',
    'bind_params',
    'CostWPO',
    'CrossFidelity',
]

import abc
import pdb
import sys
import copy

import numpy as np
import scipy as sp

import qiskit as qk
from qiskit import QiskitError
# these Classes are unfortunately now DEPRECATED by qiskit
from qiskit.aqua.operators import WeightedPauliOperator as wpo
from qiskit.aqua.operators import TPBGroupedWeightedPauliOperator as groupedwpo

# 
from qiskit.quantum_info.operators import Operator, Pauli

from . import utilities as ut

#import itertools as it
pi =np.pi

#======================#
# Basic cost interface
#======================#
class CostInterface(metaclass=abc.ABCMeta):
    """ Implements interface that can be used in batch processing"""

    def __add__(self,other):
        """ 
        '+' operator overload to allow adding of Cost objects that can
        be different subclasses.

        TODO
        ----
        - if the types of the input objects are the same the add could
          preserve that and carry forward more data
        """
        # catch whether other is a cost obj or a scalar
        if issubclass(type(other),CostInterface):
            # case: other is a cost object

            # tests for whether adding is valid
            assert self.ansatz==other.ansatz, "Cannot add two cost functions with different ansatz."
            #assert self.instance==other.instance, "Cannot add two cost functions with different quantum instances."
            
            # copy to decouple the output from the summed objects
            # currently not a deepcopy, a qiskit bug means some objs cannot be copied
            tmp_1 = copy.copy(self)
            tmp_2 = copy.copy(other)
            
            # make summed object
            sum_cost = GenericCost()
            sum_cost.ansatz = self.ansatz # allows chained operations
            sum_cost._meas_circuits = tmp_1._meas_circuits + tmp_2._meas_circuits
            sum_cost.evaluate_cost = (
                lambda x,**kwargs : tmp_1.evaluate_cost(x,**kwargs) + tmp_2.evaluate_cost(x,**kwargs)
                )
            return sum_cost

        elif (isinstance(other, (int, float, complex)) and not isinstance(other, bool)):
            # case: other is scalar

            # currently not a deepcopy, a qiskit bug means some objs cannot be copied
            sum_cost = copy.copy(self) 
            sum_cost.ansatz = self.ansatz # allows chained operations
            sum_cost.evaluate_cost = (lambda x,**kwargs : other + self.evaluate_cost(x,**kwargs))
            return sum_cost

    def __radd__(self,other):
        return self.__add__(other)

    def __mul__(self,scalar):
        """ 
        '*' operator overload to allow multiplying Cost objects with a
        scalar
        """
        # currently not a deepcopy, a qiskit bug means some objs cannot be copied
        scaled_cost = copy.copy(self)
        scaled_cost.ansatz = self.ansatz # allows chained operations
        scaled_cost.evaluate_cost = (lambda x,**kwargs : scalar*self.evaluate_cost(x,**kwargs))
        return scaled_cost

    def __rmul__(self,scalar):
        return self.__mul__(scalar)

    def __sub__(self,other):
        return self.__add__(other.__mul__(-1))

    def __rsub__(self,other):
        return self.__sub__(other)

    @property
    def meas_circuits(self):
        """ Returns list of measurement circuits needed to evaluate the cost function"""
        circs = self._meas_circuits
        return circs
    
    @property
    def qk_vars(self):
        """ Returns parameter objects in the circuit"""
        return self.ansatz.params
    
    @abc.abstractmethod
    def evaluate_cost(
        self, 
        results:qk.result.result.Result, 
        name=None,
        **kwargs,
        ):
        """ Returns the result of the cost function from a qk results object, 
            optional to specify a name to give the results list
            TODO: extend to allow list of names"""
        raise NotImplementedError
    
    def bind_params_to_meas(self, params = None, params_names = None):
        """ 
        Bind a list of parameters to named measurable circuits of the cost function 
        
        Parameters
        ----------
        params: None, or 1d, 2d numpy array (if 1d becomes 2d from np.atleast_2d)
            If None the function will return the unbound measurement circuit, 
            else it will bind each parameter to each of the measurable circuits
        params_names: if None nothing new happen (i.e. all circuits are named as self.name)
                      if not None it will prepend the params_names passedto the respective circuits, 

        Returns
        -------
            quantum circuits
                The bound or unbound named measurement circuits
        """
        if params is None:
            bound_circuits = self._meas_circuits()
        else:
            params = np.atleast_2d(params)
            if type(params_names) == str:
                params_names = [params_names]
            if params_names is None: 
                params_names = [None] * len(params)
            else:
                assert len(params_names) == len(params)
            bound_circuits = []
            for p, pn in zip(params, params_names):
                bound_circuits += bind_params(self.meas_circuits, p, self.qk_vars, pn)
        return bound_circuits   

class GenericCost(CostInterface):

    def evaluate_cost(
        self, 
        results : qk.result.result.Result, 
        name=None,
        **kwargs
        ):
        pass

#======================#
# Base class
#======================#
class Cost(CostInterface):
    """
    This base class defines all the ingredients necessary to evaluate a cost 
    function based on an ansatz circuit:
        + how should be the full(ansatz+measurements) circuit generated and 
          transpiled
        + what should be measured
        + how the measurements outcomes(counts) should be aggregated to return 
          an estimate of the cost

    Logic of computing the cost are defined by the followings (which are not
    implemented in the base class but should be implemented in the subclasses):
        + self._list_meas is a list of M strings indicating all the measurement 
            settings required
        + self._meas_func is a single function taking as an input the list of 
            the M outputs from the execution of the circuits (counts dictionaries) 
            and returning a single value

    Terminology:
        + ansatz: Object of ansatz class that has all needed properties
        + circuit: quantum circuit 
        + measurable circuit: circuit with measurement operations
        + qk_vars qiskit.parameter objects
        
    """
    def __init__(self, ansatz, instance, 
                 fix_transpile = True, # maybe redundent now
                  keep_res = False, 
                  verbose = True, 
                  debug = False, 
                  error_correction = False,
                  name = None, **args):
        """  
        Parameters
        ----------
            ansatz : Ansatz object 
            instance : qiskit quantum instance
            fix_transpile : boolean if True circuits are only transpiled once
            keep_res : boolean systematically keep the Results Object from execution
            verbose : boolean print some results
            debug: boolean allows to enter debug mode
            error_correction: not implemented yet
            name : str or None name of the circuit if None it is randomly generated
        """
        if debug: pdb.set_trace()
        if name is None:
            name = 'circuit_' + ut.gen_random_str(5)
        self.name = name
        self.ansatz = ansatz
        self.instance = instance
        self.nb_qubits = ansatz.nb_qubits  # may be redundant
        self.dim = np.power(2, ansatz.nb_qubits)
        self.nb_params = ansatz.nb_params # maybe redundant
        self.fix_transpile = fix_transpile # is it needed
        self.verbose = verbose
        self._keep_res = keep_res
        self._res = []
        # These methods needs to be implemented in the subclasses
        #--------------------------------------
        self._list_meas = self._gen_list_meas()  
        self._meas_func = self._gen_meas_func() 
        #--------------------------------------
        self.main_circuit = ansatz.circuit
        self._untranspiled_main_circuit = copy.deepcopy(ansatz.circuit)
        self._qk_vars = ansatz.params
        self._meas_circuits = gen_meas_circuits(self._untranspiled_main_circuit, 
                                                self._list_meas)
        self._meas_circuits = self.instance.transpile(self._meas_circuits)
        self._label_circuits()
        #--------------------------------------
        self.err_corr = error_correction
        if(self.err_corr):
            raise NotImplementedError
            
        if(args.get('invert')):
            self._wrap_cost = lambda x: 1-x
        else:
            self._wrap_cost = lambda x: x
    
    def __call__(self, params, debug=False):
        """ Estimate the CostFunction for some parameters
        Has a known bug: if number of measurement settings > max_job_size  
        Parameters
        ----------
        params :either 1d, 2d of numpy array (if 1d becomes 2d from np.atleast_2d)
        debug: boolean allows to enter debug mode

        Returns
        -------
            res: 2d array 
        """
        if debug: pdb.set_trace()
        params = np.atleast_2d(params)
        name_params = ['x' + str(i) + 'x' for i in range(len(params))]

        # List of all the circuits to be ran
        bound_circs = self.bind_params_to_meas(params, name_params)

        # Execute them
        results = self.instance.execute(bound_circs, 
                                        had_transpiled=self.fix_transpile)  
        if self._keep_res:
            self._res.append(results)
        # Evaluate cost functions based on results
        res = np.array(self.evaluate_cost(results, name = name_params))
            
        if np.ndim(res) == 1: 
            res = res[:,np.newaxis]
        if self.verbose: print(res)
        return res 


    def _gen_qk_vars(self):
        raise NotImplementedError("This function is now in the Ansatz class")        
                    
    def _init_res(self):
        """ Flush the res accumulated so far """
        self._res = []

    def _gen_list_meas(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError()
        
    def _gen_meas_func(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError()
    
    def _label_circuits(self):
        """ Give names to all circuits to they can be identified in the results obj"""
        self.main_circuit.name = self.name
        for c in self._meas_circuits:
            c.name = self.name
    
    def evaluate_cost(self, results_obj, name=None, **kwargs):
        """ Returns cost value from a qiskit result object
        ----------
        results_obj : None, or 1d, 2d numpy array (if 1d becomes 2d from np.atleast_2d)
            If None the function will return the unbound measurement circuit, 
            else it will bind each parameter to each of the measurable circuits
        name: if None it will use self.name instead
              if str it will filter the result obj with a matching name (i.e. name is included) 
              if list<str> same behavior but for each element of the list

        Returns
        -------
            cost: scalar-like value if name is None or string
                  list of costs
        Assume the right ordering in results_obj
        """
        if name == None:
            name = self.name
        if type(name) is str:
            res = self._evaluate_cost_one_name(results_obj, name)
        else:
            res = [self._evaluate_cost_one_name(results_obj, n) for n in name]
        return res

    def _evaluate_cost_one_name(self, results_obj, name):
        """ same as above except it takes as input a single name"""
        count_list = []
        for ii in range(len(results_obj.results)):
            if name in results_obj.results[ii].header.name:
                count_list.append(results_obj.get_counts(ii))
        return self._wrap_cost(self._meas_func(count_list))

    def shot_noise(self, params, nb_experiments=8):
        """ Sends a single job many times to see shot noise"""        
        params = [params for ii in range(nb_experiments)]
        return self.__call__(params)
    

    # Comparison/informations about the circuits
    def check_layout(self):
        """ Draft, check if all the meas_circuit have the same layout
        TODO: remove except if really needed
        """
        ref = self.main_circuit
        test = [compare_layout(ref, c) for c in self._meas_circuits]
        return np.all(test)

    def compare_layout(self, cost2, verbose=True):
        """ Draft, goal compare transpiled circuits (self._maincircuit)
        and ensure they have the same layout"""
        test1 = self.check_layout()
        test2 = cost2.check_layout()
        test3 = compare_layout(self.main_circuit, cost2.main_circuit)
        if verbose: 
            print("self: same layout - {}".format(test1))
            print("cost2: same layout - {}".format(test2))
            print("self and cost2: same layout - {}".format(test3))
        return test1 * test2 *test3
    
    def check_depth(self, long_output=False, delta = 1):
        """ Check the depths of the measurable circuits, are all within a delta
        """
        depth = [c.depth() for c in self._meas_circuits]
        test = (max(depth) - min(depth)) <=delta
        return test
    
    def get_depth(self, num=None):
        """ Get the depth of the circuit(s)
        if num=None main_circuit / num=-1 all the meas_circ / else meas_circ[num] 
        """
        circ = self._return_circuit(num)
        depth = [c.depth() for c in circ]
        return depth
    
    def compare_depth(self, cost2, verbose=True, delta=0):
        """ Draft, goal compare transpiled circuits (self._maincircuit)
        and ensure they have the same layout"""
        depth1 = self.check_depth(long_output=True)
        depth2 = cost2.check_depth(long_output=True)
        test1 = np.abs(max(depth1) - max(depth2)) <= delta
        test2 = np.abs(min(depth1) - min(depth2)) <= delta
        test = test1 and test2
        if verbose: 
            print("self and cost2: same depth - {}".format(test))
            print("self min-max: {} and {}".format(min(depth1), max(depth1)))
            print("cost2 min-max: {} and {}".format(min(depth2), max(depth2)))
        return test

    def draw(self, num=None, depth = False):
        """ Draw one of the circuit 
        if num=None main_circuit / num=-1 all the meas_circ / else meas_circ[num] 
        """
        circs = self._return_circuit(num)
        for c in circs:
            print(c)
            if depth:
                print(c.depth())
        
    def _return_circuit(self, num=None):
        """ Return a list of circuits according to num following the convention:
        if num=None main_circuit / num=-1 all the meas_circ / else meas_circ[num] 
        """
        if num is None:
            circ = [self.main_circuit]
        elif num >= 0:
            circ = [self._meas_circuits[num]]
        elif num == -1:
            circ = self._meas_circuits
        return circ


#======================#
# Subclasses: one-qubit related costs
#======================#
class OneQProjZ(Cost):
    """ Fidelity w.r.t. to the target state |1> """   
    def _gen_list_meas(self):
        """ 1 measurement settings"""
        return ['z']
    
    def _gen_meas_func(self):
        """ return the frequency of counts"""
        def meas_func(counts):
            return 1 - freq_even(counts[0]) 
        return meas_func


class OneQXYZ(Cost):
    """ Fidelity w.r.t. to a 1qubit target state specified in terms of its 
    decomposition in the Pauli basis"""   
    def __init__(self, ansatz, instance, coeffs = None, decompose = False, 
        fix_transpile = True, keep_res = True, verbose = True, debug = False, 
        error_correction = False):
        """ 
        coeffs: list of size 3 with elements [a_x = <X>tgt, a_y = <Y>tgt, a_z = <Z>tgt] 
                s.t psi_tgt =  1/2 (a_x X + a_y Y +a_z Z)
                if None is passed they will be randomly generated
                In both case they are normalized (i.e. the target state is pure)
        decompose: if True the estimated fidelity is returned when calling the object
                   if False return an array with the 3 frequencies
        """
        if coeffs is None:
            coeffs = np.random.uniform(size=3)
        coeffs /= np.dot(coeffs, coeffs)
        self.coeffs = coeffs
        self.decompose = decompose
        super(OneQXYZ, self).__init__(ansatz,  instance, fix_transpile, keep_res, 
                            verbose, debug, error_correction)

    def _gen_list_meas(self):
        """ 3 measurement settings"""
        return ['x','y','z']
    
    def _gen_meas_func(self):
        """ depending on the state of self.decompose will return a function outputing
        either the weighted sum of the expectation values (decompose=False) or 
        an array with the estimated frequencies"""
        weights = self.coeffs
        dim = self.dim
        if self.decompose:
            def meas_func(counts):
                return [freq_even(c) for c in counts]
        else:
            def meas_func(counts):
                return (1+np.dot([expected_parity(c) for c in counts], weights))/dim
        return meas_func
    

#======================#
# Subclasses: GHZ related costs
#======================#
class GHZPauliCost(Cost):
    """ Cost = fidelity w.r.t. a N-qubit GHZ state, estimated based on the 
    expected values of N-fold Pauli operators (e.g. 'XXY')
    """   
    # Hardcoded list of measurements settings for GHZ of different sizes
    # {'nb_qubits':(meas_strings, weights)}, wehere the measurement string is a 
    # list of Pauli operators and the weights correspond to the decomposition 
    # of the GHZ state in the Pauli tensor basis (up to a constant 1/dim)
    # It could be automated to deal with arbitrary size state
    _GHZ_PAULI_DECOMP = {
    '2':(
            ['xx', 'yy', 'zz'], 
            np.array([1.,-1.,1.])
            ),
    '3':(
            ['1zz','xxx','xyy','yxy','yyx','z1z','zz1'], 
            np.array([1., 1., -1., -1., -1., 1.,1.])
            ),
    '4':( 
            ['11zz','1z1z','1zz1','xxxx','xxyy','xyxy','xyyx','yxxy','yxyx',
             'yyxx','yyyy','z11z','z1z1','zz11','zzzz'],
            np.array([1.,1.,1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,1.,1.,1.,1.,1.])
            )
        }
        
    def _gen_list_meas(self):
        return self._GHZ_PAULI_DECOMP[str(self.nb_qubits)][0]
    
    def _gen_meas_func(self):
        """ expected parity associated to each of the measurement settings"""
        weights = self._GHZ_PAULI_DECOMP[str(self.nb_qubits)][1]
        dim = self.dim
        def meas_func(counts):
            return (1+np.dot([expected_parity(c) for c in counts], weights))/dim
        return meas_func

class GHZWitness1Cost(Cost):
    """ Cost based on witnesses for genuine entanglement ([guhne2005])
    Stabilizer generators S_l of GHZ are (for n=4) S = <XXXX, ZZII, IZZI, IIZZ>
    To estimate S_1 to S_n only requires two measurement settings: XXXX, ZZZZ
    Cost =  (S_1 - 1)/2 + Prod_l>1 [(S_l + 1)/2] """   
    def _gen_list_meas(self):
        """ two measurement settings ['x...x', 'z...z']"""
        N = self.nb_qubits
        list_meas = ['x'*N, 'z'*N]
        return list_meas
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        def meas_func(counts):
            S1 = freq_even(counts[0])
            S2 = np.array([freq_even(counts[1], indices=[i,i+1]) for i in range(N-1)])
            return 0.5*(S1-1) + np.prod((S2+1)/2)
        return meas_func

class GHZWitness2Cost(Cost):
    """ Exactly as GHZWitness1Cost except that Cost =  Sum_l[S_l] - (N-1)I """   
    
    def _gen_list_meas(self):
        """ two measurement settings ['x...x', 'z...z']"""
        N = self.nb_qubits
        list_meas = ['x'*N, 'z'*N]
        return list_meas
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        def meas_func(counts):
            S1 = freq_even(counts[0])
            S2 = np.array([freq_even(counts[1], indices=[i,i+1]) for i in range(N-1)])
            return S1 + np.sum(S2) - (N -1)
        return meas_func
    
#======================#
# Subclasses: Graph states
#======================#    
class GraphCyclPauliCost(Cost):
    """ A N-qubit Cyclical graph has edges = [[1,2],[2,3],...,[N-1,N],[N,1]]
    Cost = fidelity, estimated based on the expected values of the N-fold Pauli 
    operators (e.g. 'XXY')
    """   
    # Hardcoded list of measurements settings for Cyclical graph states of 
    #different sizes {'nb_qubits':(meas_strings, weights)}, wehere the measurement 
    # string is a list of Pauli operators and the weights correspond to the 
    # decomposition of the target state in the Pauli tensor basis (up to a constant 1/dim)
    # It could be automated to deal with arbitrary size state
    _CYCLICAL_PAULI_DECOMP = {
    '2':(
            ['1x','x1','xx'], 
            np.array([1,1,1])
            ),
    '3':(
            ['1yy','xxx','xzz','y1y','yy1','zxz','zzx'], 
            np.array([1,-1,1,1,1,1,1])
            ),
    '4':( 
            ['1x1x','1yxy','1zxz','x1x1','xxxx','xy1y','xz1z','y1yx','yxy1','yyzz',
             'yzzy','z1zx','zxz1','zyyz','zzyy'],
            np.array([1,-1,1,1,1,-1,1,-1,-1,1,1,1,1,1,1])
            ),
    '5':( 
            ['11zxz','1x1yy','1xzzx','1yxxy','1yy1x','1zxz1','1zyyz','x1xzz','x1yy1',
             'xxxxx','xxy1y','xy1yx','xyzzy','xz11z','xzzx1','y1x1y','y1yxx','yxxy1',
             'yxyzz','yy1x1','yyz1z','yz1zy','yzzyx','z11zx','z1zyy','zx1xz','zxz11',
             'zyxyz','zyyz1','zzx1x','zzyxy'],
            np.array([1,1,1,1,1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,1,-1,1,1,-1])
            ),
    '6':( 
            ['111zxz','11zxz1','11zyyz','1x1x1x','1x1yxy','1xz1zx','1xzzyy','1yxxxy',
             '1yxy1x','1yy1yy','1yyzzx','1zx1xz','1zxz11','1zyxyz','1zyyz1','x1x1x1',
             'x1xz1z','x1yxy1','x1yyzz','xxxxxx','xxxy1y','xxy1yx','xxyzzy','xy1x1y',
             'xy1yxx','xyz1zy','xyzzyx','xz111z','xz1zx1','xzzxzz','xzzyy1','y1x1yx',
             'y1xzzy','y1yxxx','y1yy1y','yxxxy1','yxxyzz','yxy1x1','yxyz1z','yy1xzz',
             'yy1yy1','yyz11z','yyzzx1','yz11zy','yz1zyx','yzzx1y','yzzyxx','z111zx',
             'z11zyy','z1zx1x','z1zyxy','zx1xz1','zx1yyz','zxz111','zxzzxz','zyxxyz',
             'zyxyz1','zyy1xz','zyyz11','zzx1yy','zzxzzx','zzyxxy','zzyy1x'],
            np.array([1,1,1,1,-1,1,1,-1,-1,1,1,1,1,-1,1,1,1,-1,1,1,-1,-1,1,-1,-1,
                      -1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,-1,1,1,1,1,1,-1,1,1,1,1,1,
                      -1,1,1,1,1,1,-1,1,1,1,1,1,1])
            )
        }
        
    def _gen_list_meas(self):
        return self._CYCLICAL_PAULI_DECOMP[str(self.nb_qubits)][0]
    
    def _gen_meas_func(self):
        """ expected parity associated to each of the measurement settings"""
        weights = self._CYCLICAL_PAULI_DECOMP[str(self.nb_qubits)][1]
        dim = self.dim
        def meas_func(counts):
            return (1+np.dot([expected_parity(c) for c in counts], weights))/dim
        return meas_func

class GraphCyclWitness1Cost(Cost):
    """ Cost function based on the construction of witnesses for genuine 
    entanglement ([guhne2005])
    Stabilizer generators S_l of cyclical graph states are (for N=4 qubits) 
        S = <XZIZ, ZXZI, IZXZ, ZIZX>
    To estimate S_1 to S_N only requires two measurement settings: XZXZ, ZXZX
    Cost =  (S_1 - 1)/2 + Prod_l>1 [(S_l + 1)/2] 
    !!! ONLY WORK FOR EVEN N FOR NOW !!!
    !!! PROBABLY WRONG (or at least not understood clearly) !!!
    """
    def _gen_list_meas(self):
        """ two measurement settings ['zxzx...zxz', 'xzxzx...xzx']"""
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            meas_odd = "".join(['zx'] * (N//2))
            meas_even = "".join(['xz'] * (N//2))
        return [meas_odd, meas_even]
    
    def _gen_meas_func(self):
        raise Warning("This is likely broken be careful")
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            ind_odd = [[i, i+1, i+2] for i in range(0,N-2, 2)] + [[0, N-2, N-1]]  
            ind_even = [[i, i+1, i+2] for i in range(1,N-2, 2)] + [[0, 1, N-1]]
            def meas_func(counts):
                counts_odd, counts_even = counts[0], counts[1]
                S_odd = np.array([expected_parity(counts_odd, indices=i) for i in ind_odd])
                S_even = np.array([expected_parity(counts_even, indices=i) for i in ind_even])
                return 0.5*(S_even[-1]-1) + np.prod((S_odd+1)/2) * np.prod((S_even[:-1]+1)/2) 
        return meas_func

class GraphCyclWitness2Cost(Cost):
    """ Exactly as GraphCyclWitness1Cost except that:
        Cost =  Sum_l[S_l] - (N-1)I """   
    def _gen_list_meas(self):
        """ two measurement settings ['zxzx...zxz', 'xzxzx...xzx']"""
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            meas1 = "".join(['zx'] * (N//2))
            meas2 = "".join(['xz'] * (N//2))
        return [meas1, meas2]
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            ind_odd = [[i, i+1, i+2] for i in range(0,N-2, 2)] + [[0, N-2, N-1]]  
            ind_even = [[i, i+1, i+2] for i in range(1,N-2, 2)] + [[0, 1, N-1]]
            def meas_func(counts):
                counts_odd, counts_even = counts[0], counts[1]
                S_odd = np.array([expected_parity(counts_odd, indices=i) for i in ind_odd])
                S_even = np.array([expected_parity(counts_even, indices=i) for i in ind_even])
                return np.sum(S_odd) + np.sum(S_even) - (N-1)
        return meas_func

class GraphCyclWitness2FullCost(Cost):
    """ Same cost function as GraphCyclWitness2Cost, except that the measurement
    settings to obtain the expected values of the generators S_l have been
    splitted into N measurent settings (rather than 2), and now each measurement
    settings involved only 3 measurements instead of N
    -> measurement outcomes should be less noisy as less measurements are
       involved per measurement settings
    """   
    def _gen_list_meas(self):
        """ N measurement settings ['xz1..1z', 'zxz1..1', .., 'z1..1zx' ]"""
        N = self.nb_qubits
        list_meas = []
        for ind in range(N):    
            meas = ['1'] * N
            meas[(ind-1) % N] = 'z'
            meas[ind % N] = 'x'
            meas[(ind+1) % N] = 'z'
            list_meas.append(''.join(meas))
        return list_meas
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        def meas_func(counts):
            exp = [expected_parity(c) for c in counts]
            return np.sum(exp)  - (N-1)
        return meas_func

class GraphCyclWitness3Cost(Cost):
    """ Exactly as GraphCyclWitness1Cost except that Cost =  XXX
    To implement"""   
    
    def _gen_list_meas(self):
        """ N measurement settings ['xz1..1z', 'zxz1..1', .., 'z1..1zx' ]"""
        N = self.nb_qubits
        list_meas = []
        for ind in range(N):    
            meas = ['1'] * N
            meas[(ind-1) % N] = 'z'
            meas[ind % N] = 'x'
            meas[(ind+1) % N] = 'z'
            list_meas.append(''.join(meas))
        return list_meas
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        def meas_func(counts):
            exp = [expected_parity(c) for c in counts]
            return np.sum(exp)  - (N-1)
        return meas_func

#======================#
# Random xy-Hamiltonian related cost
#======================#
class RandomXYCost(Cost):
    """
    Cost function for energy expectation value of the random 1D xy hamiltonian
    
    Custom parameters
    -----------
    hamiltonian : 2D np array
        Diagonal elements are the longitudinal (z) field terms.
        Off diagonal elements are the random couplings of the (XX + YY) terms
    """
    def __init__(self, ansatz, instance, hamiltonian,
                 fix_transpile = True, # maybe redundent now
                 keep_res = False, 
                 verbose = True, 
                 debug = False, 
                 error_correction = False,
                 name = None, **args):
        super().__init__( ansatz, instance, 
                         fix_transpile, # maybe redundent now
                         keep_res, 
                         verbose, 
                         debug, 
                         error_correction,
                         name, **args)
        assert self.nb_qubits == hamiltonian.shape[0], "Input hamiltonian must have same dims as nb_qubits (see docstring)"
        assert hamiltonian.shape[0] == hamiltonian.shape[1], "Input Hamiltonian should be square (see docstring)"
        self.hamiltonian = hamiltonian
    def _gen_list_meas(self):
        nb_qubits = self.nb_qubits
        z = 'z'*nb_qubits
        x = 'x'*nb_qubits
        y = 'y'*nb_qubits
        return [z,x,y]
    
    def _gen_meas_func(self):
        def func(count_list):
            longitudinal_field = np.diag(self.hamiltonian)
            field_term = [longitudinal_field[ii] * ut.pauli_correlation(count_list[0], ii) for ii in range(self.nb_qubits)]
            field_term = sum(field_term)
            xy_term = 0
            for ii in range(self.nb_qubits):
                for jj in range(self.nb_qubits):
                    if ii != jj:
                        xy_term += self.hamiltonian[ii,jj] * ut.pauli_correlation(count_list[1], ii, jj)
                        xy_term += self.hamiltonian[ii,jj] * ut.pauli_correlation(count_list[2], ii, jj)
            return field_term + xy_term
        return func

# ------------------------------------------------------
# Functions to compute expected values based on measurement outcomes counts as 
# returned by qiskit
# ------------------------------------------------------
def freq_even(count_result, indices=None):
    """ return the frequency of +1 eigenvalues:
    The +1 e.v. case corresponds to the case where the number of 0 in the 
    outcome string is even
    
    indices: list<integer>
             if not None it allows to consider only selected elements of the 
             outcome string
    """
    nb_odd, nb_even = 0, 0
    for k, v in count_result.items():
        k_invert = k[::-1]
        sub_k = get_substring(k_invert, indices)
        nb_even += v * (sub_k.count('1')%2 == 0)
        nb_odd += v * (sub_k.count('1')%2)
    return nb_even / (nb_odd + nb_even)

def expected_parity(results,indices=None):
    """ return the estimated value of the expectation of the parity operator:
    P = P+ - P- where P+(-) is the projector 
    Comment: Parity operator ircuit.quantumcircuit.QuantumCircuitircuit.quantumcircuit.QuantumCircuitmay nor be the right name
    """
    return 2 * freq_even(results, indices=indices) - 1

def get_substring(string, list_indices=None):
    """ return a substring comprised of only the elements associated to the 
    list of indices
    Comment: probably already exist or there may be a better way"""
    if list_indices == None:
        return string
    else:
        return "".join([string[ind] for ind in list_indices])

# ------------------------------------------------------
# Some functions to deals with appending measurement, param bindings and comparisons
# ------------------------------------------------------
def compare_layout(circ1, circ2):
    """ Draft, define a list of checks to compare transpiled circuits
        not clear what the rules should be (or what would be a better name)
        So far: compare the full layout"""
    test = True
    test &= (circ1._layout.get_physical_bits() == circ2._layout.get_physical_bits())
    test &= (circ1.count_ops()['cx'] == circ2.count_ops()['cx'])
    return test

def append_measurements(circuit, measurements, logical_qubits=None):
    """ Append measurements to one circuit:
        TODO: Replace with Weighted pauli ops?"""
    circ = copy.deepcopy(circuit)
    num_creg = len(measurements.replace('1',''))
    if num_creg > 0:
        cr = qk.ClassicalRegister(num_creg, 'classical')
        circ.add_register(cr)
    if logical_qubits is None: 
        logical_qubits = np.arange(circ.num_qubits)
    creg_idx = 0
    for qb_idx, basis in enumerate(measurements):
        qubit_number = logical_qubits[qb_idx]
        if basis == 'z':
            circ.measure(qubit_number, creg_idx)
            creg_idx += 1
        elif basis == 'x':
            circ.u2(0.0, pi, qubit_number)  # h
            circ.measure(qubit_number, creg_idx)
            creg_idx += 1
        elif basis == 'y':
            circ.u1(-np.pi / 2, qubit_number)  # sdg
            circ.u2(0.0, pi, qubit_number)  # h
            circ.measure(qubit_number, creg_idx)
            creg_idx += 1
        elif basis != '1':
            raise NotImplementedError('measurement basis {} not understood').format(basis)
    return circ

def gen_meas_circuits(main_circuit, meas_settings, logical_qubits=None):
    """ Return a list of measurable circuit based on a main circuit and
    different settings"""
    c_list = [append_measurements(main_circuit.copy(), m, logical_qubits) 
                  for m in meas_settings] 
    return c_list

def bind_params(circ, param_values, param_variables, param_name = None):
    """ Take a list of circuits with bindable parameters and bind the values 
    passed according to the param_variables
    Returns the list of circuits with bound values 
    DOES NOT MODIFY INPUT (i.e. hardware details??)
    Parameters
    ----------
    circ : single or list of quantum circuits with the same qk_vars
    params_values: a 1d array of parameters (i.e. correspond to a single 
        set of parameters)
    param_variables: list of qk_vars, it should match element-wise
        to the param_values
    param_name: str if not None it will used to prepend the names
        of the circuits created

    Returns
    -------
        quantum circuits
    """
    if type(circ) != list: circ = [circ]
    val_dict = {key:val for key,val in zip(param_variables, param_values)}
    bound_circ = [cc.bind_parameters(val_dict) for cc in circ]
    if param_name is not None:
        bound_circ = ut.prefix_to_names(bound_circ, param_name)
    return bound_circ  

#======================#
# Qiskit WPO class
#======================#

class CostWPO(CostInterface):
    """
    Cost class that internally uses the qiskit weighted product operator
    objects. NOTE: WeightedPauliOperator is DEPRECATED in qiskit.
    """
    def __init__(
        self,
        ansatz,
        instance,
        weighted_pauli_operators,
        ):
        """
        Parameters
        ----------
        ansatz : object implementing AnsatzInterface
            The ansatz object that this cost can be optimsed over
        instance : qiskit quantum instance
            Will be used to generate internal transpiled circuits
        weighted_pauli_operators : qiskit WeightedPauliOperator
            Pauli operators whose weighted sum defines the cost
        """
        self.ansatz = ansatz
        self.instance = instance

        # check type of passed operators
        if not type(weighted_pauli_operators) is wpo:
            raise TypeError

        # ensure the ansatz and qubit Hamiltonians have same number of qubits
        assert weighted_pauli_operators.num_qubits==self.ansatz.nb_qubits

        # store operators in grouped form, currently use `unsorted_grouping` method, which
        # is a greedy method. Sorting method could be controlled with a kwarg
        self.grouped_weighted_operators = groupedwpo.unsorted_grouping(weighted_pauli_operators)

        # generate and transpile measurement circuits
        measurement_circuits = self.grouped_weighted_operators.construct_evaluation_circuit(
            wave_function=copy.deepcopy(self.ansatz.circuit),
            statevector_mode=self.instance.is_statevector
            )
        self._meas_circuits = self.instance.transpile(measurement_circuits)

    def evaluate_cost_and_std(
        self, 
        results:qk.result.result.Result, 
        name='',
        ):
        """ 
        Evaluate the expectation value of the state produced by the 
        ansatz against the weighted Pauli operators stored, using the
        results from an experiment.

        NOTE: this takes the statevector mode from the cost obj's quantum
        instance attribute, which is not necessarily the instance that 
        has produced the results. The executing instance should be the
        same backend however, and so (hopefully) operate in the same 
        statevector mode.

        Parameters
        ----------
        results : qiskit results obj
            Results to evaluate the operators against
        name : string, optional
            Used to resolve circuit naming
        """
        return self.grouped_weighted_operators.evaluate_with_result(
            results,
            statevector_mode=self.instance.is_statevector,
            circuit_name_prefix=name
            )

    def evaluate_cost(
        self, 
        results:qk.result.result.Result, 
        name='',
        real_part=True,
        **kwargs,
        ):
        """ 
        Evaluate the expectation value of the state produced by the 
        ansatz against the weighted Pauli operators stored, using the
        results from an experiment.

        Parameters
        ----------
        results : qiskit results obj
            Results to evaluate the operators against
        name : string, optional
            Used to resolve circuit naming
        """
        mean,std = self.evaluate_cost_and_std(
            results=results, 
            name=name
            )
        if real_part:
            return np.real(mean)
        else:
            return mean

#======================#
# Cross-fidelity class
#======================#

class CrossFidelity(CostInterface):
    """
    Cost class to implement offline CrossFidelity measurements between 
    two quantum states (arxiv:1909.01282)
    """
    def __init__(
        self,
        ansatz,
        instance,
        comparison_results=None,
        seed=0,
        nb_random=5,
        prefix='HaarRandom',
        ):
        """
        Parameters
        ----------
        ansatz : object implementing AnsatzInterface
            The ansatz object that this cost can be optimsed over
        instance : qiskit quantum instance
            Will be used to generate internal transpiled circuits
        comparison_results : {dict, None}
            The use cases where None would be passed is if we are using
            this object to generate the comparison_results object for a
            future instance of CrossFidelity. This robustly ensures that
            the results objs to compare are compatible.
            If dict is passed it should be a qiskit results object that
            has been converted to a dict using its `to_dict` method. 
            Ideally this would have been tagged with CrossFidelity 
            metadata using this classes `tag_results_metadata` method.
        seed : int, optional
            Seed used to generate random unitaries
        nb_random : int, optional
            The number of random unitaries to average over
        prefix : string, optional
            String to use to label the measurement circuits generated
        """

        # store inputs
        self.ansatz = ansatz
        self.instance = instance

        # store hidden properties
        self._nb_random = nb_random
        assert isinstance(self._nb_random,int) and (self._nb_random>0), 'nb_random is invalid.'
        self._prefix = prefix
        self._seed = seed

        # generate and store set of measurement circuits here
        self._meas_circuits = self._gen_random_measurements()
        self._meas_circuits = self.instance.transpile(self._meas_circuits)

        # run setter (see below)
        self.comparison_results = comparison_results

    @property
    def nb_random(self):
        return self._nb_random
    
    @property
    def seed(self):
        return self._seed

    @property
    def comparison_results(self):
        return self._comparison_results

    @comparison_results.setter
    def comparison_results(self, results):
        """ 
        setter for comparison_results, perform validations
        """
        # check if comparison_results contains the crossfidelity_metadata
        # tags and if it does compare them, if these comparisons fail then
        # crash, if the crossfidelity_metadata is missing issue a warning
        if not results is None:
            if not type(results) is dict:
                results = results.to_dict()

            comparison_metadata = None
            try:
                comparison_metadata = results['crossfidelity_metadata']
            except KeyError:
                print('Warning, input results dictionary does not contain crossfidelity_metadata'
                    +' and so we cannot confirm that the results are compatible. If the input results'
                    +' object was collecting by this class consider using the tag_results_metadata'
                    +' method to add the crossfidelity_metadata.',file=sys.stderr)
            if not comparison_metadata is None:
                _err_msg = ('Input results dictionary contains data that is incompatible with the '
                        +' this CrossFidelity object.')
                assert self._seed == comparison_metadata['seed'],_err_msg
                assert (not self._nb_random > comparison_metadata['nb_random']),_err_msg
                assert self._prefix == comparison_metadata['prefix'],_err_msg

            # bug fix, need counts dict keys to be hex values
            for idx,v in enumerate(results['results']):
                new_counts = {}
                for ck,cv in v['data']['counts'].items():
                    # detect it is not hex
                    if not ck[:2]=='0x':
                        ck = hex(int(ck,2))
                    new_counts[ck] = cv
                v['data']['counts'] = new_counts
    
        self._comparison_results = results

    def _gen_random_measurements(self):
        """ 
        Creates a list of self._nb_random circuits with Haar random 
        unitaries appended to measure in random basis for each qubit

        Returns
        -------
            quantum circuits
                The unbound and untranspiled circuits to carry out 
                different random measurements on each qubit
        """
        
        # random state object used to generate random unitaries, using the
        # same seed means multiple calls to this function will produce the 
        # same set of random measurements
        rand_state = np.random.RandomState(seed=self._seed)

        circ_list = []
        for ii in range(self._nb_random):

            # make random measurment circuit
            rand_measurement = qk.QuantumCircuit(self.ansatz.nb_qubits)
            for i in range(self.ansatz.nb_qubits):
                R = qk.quantum_info.random_unitary(2,seed=rand_state)
                rand_measurement.append(R,[i])
            rand_measurement.measure_all()

            # append to ansatz circuit and rename
            circ = self.ansatz.circuit + rand_measurement
            circ.name = self._prefix+str(ii)
            circ_list.append(circ)

        return circ_list

    def tag_results_metadata(self,results):
        """
        Adds in CrossFidelity metadata to a results object. This can be
        used to ensure that two results sets are compatible.

        Parameters
        ----------
        results : Qiskit results type, or dict
            The results data to process

        Returns
        -------
        results : dict
            Results dictionary with the CrossFidelity metadata added
        """
        # convert results to dict if needed
        if not type(results) is dict:
            results = results.to_dict()
        # add CrossFidelity metadata
        results.update({
            'crossfidelity_metadata':{
                'seed':self._seed,
                'nb_random':self._nb_random,
                'prefix':self._prefix,
                }
            })
        return results

    def evaluate_cost(
        self,
        results,
        nb_random=None,
        name='',
        **kwargs
        ):
        """ 
        Calculates the cross-fidelity using two sets of qiskit results. 
        The variable names are chosen to match arxiv:1909.01282 as close
        as possible.

        Parameters
        ----------
        results : Qiskit results type
            Results to calculate cross-fidelity with, against the stored
            results dictionary.
        nb_random : int, optional
            The number of random unitaries to use to compute the cost. 
            This allows subsampling of the full amount of data available
            in the results.

        Returns
        -------
        cross_fidelity : float
            Evaluated cross-fidelity
        """

        # we make it possible to instance a CrossFidelity obj without a
        # comparison_results dict so that we can easily generate the 
        # comparison data using the same setup (e.g. seed, prefix). But
        # in that case cannote evaluate the cost.
        if self._comparison_results is None:
            print('No comparison results set has been passed to CrossFidelity obj.',
                file=sys.stderr)
            raise ValueError

        # convert comparison_results back to qiskit results obj, so we can
        # use `get_counts` method
        comparison_results = qk.result.Result.from_dict(self._comparison_results)

        # (optional) use less than the full number of random unitaries available
        _nb_random = self._nb_random
        if not nb_random is None:
            assert (not nb_random>self._nb_random), ("If nb_random is explicitly"
                +" declared in CrossFidelity.evaluate_cost it cannot be greater"
                +" than the objects nb_random attribute.")
            _nb_random = nb_random

        # iterate over the different random unitaries
        tr_rho1_rho2 = 0.
        tr_rho1squared = 0.
        tr_rho2squared = 0.
        nb_qubits = None
        for uidx in range(_nb_random):

            # try to extract matching experiment data
            try:
                countsdict_1_fixedU = results.get_counts(name+self._prefix+str(uidx))
                countsdict_2_fixedU = comparison_results.get_counts(self._prefix+str(uidx))
            except QiskitError:
                print('Cannot extract matching experiment data to calculate cross-fidelity.',
                    file=sys.stderr)
                raise

            # normalise counts dict to give empirical probability dists
            P_1_fixedU = { k:v/sum(countsdict_1_fixedU.values()) for k,v in countsdict_1_fixedU.items() }
            P_2_fixedU = { k:v/sum(countsdict_2_fixedU.values()) for k,v in countsdict_2_fixedU.items() }

            # use this to check number of qubits has been consistent
            # over all random unitaries
            if nb_qubits is None:
                # get the first dict key string and find its length
                nb_qubits = len(list(P_1_fixedU.keys())[0])    
            assert nb_qubits==len(list(P_1_fixedU.keys())[0]),('nb_qubits='+f'{nb_qubits}'
                +', P_1_fixedU.keys()='+f'{P_1_fixedU.keys()}')
            assert nb_qubits==len(list(P_2_fixedU.keys())[0]),('nb_qubits='+f'{nb_qubits}'
                +', P_2_fixedU.keys()='+f'{P_2_fixedU.keys()}')

            tr_rho1_rho2 += self.correlation_fixed_U(P_1_fixedU,P_2_fixedU)
            tr_rho1squared += self.correlation_fixed_U(P_1_fixedU,P_1_fixedU)
            tr_rho2squared += self.correlation_fixed_U(P_2_fixedU,P_2_fixedU)

        # add final normalisations
        tr_rho1_rho2 = (2**nb_qubits)*tr_rho1_rho2/(_nb_random)
        tr_rho1squared = (2**nb_qubits)*tr_rho1squared/(_nb_random)
        tr_rho2squared = (2**nb_qubits)*tr_rho2squared/(_nb_random)

        return tr_rho1_rho2 / max(tr_rho1squared,tr_rho2squared)

    @staticmethod
    def correlation_fixed_U(P_1,P_2):
        """ 
        Carries out the inner loop calculation of the Cross-Fidelity. In
        contrast to the paper, arxiv:1909.01282, it makes sense for us to
        make the sum over sA and sA' the inner loop. So this computes the
        sum over sA and sA' for fixed random U.

        Parameters
        ----------
        P_1 : dict (normalised counts dictionary)
            The empirical distribution for the measurments on qubit 1
            P^{(1)}_U(s_A) = Tr[ U_A \rho_1 U^\dagger_A |s_A\rangle \langle s_A| ]
            where U is a fixed, randomly chosen unitary, and s_A is all 
            possible binary strings in the computational basis
        P_2 : dict (normalised counts dictionary)
            Same for qubit 2.

        Return
        ------
        correlation_fixed_U : float
            Evaluation of the inner sum of the cross-fidelity
        """
        # iterate over the elements of the computational basis (that 
        # appear in the measurement results)sublimes
        correlation_fixed_U = 0
        for sA,P_1_sA in P_1.items():
            for sAprime,P_2_sAprime in P_2.items():

                # add up contribution
                hamming_distance = int(len(sA)*sp.spatial.distance.hamming(list(sA), list(sAprime)))                    
                correlation_fixed_U += (-2)**(-hamming_distance) * P_1_sA*P_2_sAprime

        return correlation_fixed_U
        
#%%
# -------------------------------------------------------------- #
if __name__ == '__main__':
    #from qiskit.test.mock import FakeRochester
    #fake = FakeRochester() # not working
    import ansatz as anz
    simulator = qk.Aer.get_backend('qasm_simulator')
    inst = qk.aqua.QuantumInstance(simulator, shots=8192, optimization_level=3)
    backends = [simulator]
    
    for sim in backends:
        #-----#
        # Verif conventions
        #-----#
        fun_ansatz = anz._GHZ_3qubits_6_params_cx0
        ansatz = anz.AnsatzFromFunction(fun_ansatz)
        bound_circ = bind_params(ansatz.circuit, [1,2,3,4,5,6], ansatz.circuit.parameters)
        
        transpiled_cir = inst.transpile(bound_circ)[0]
        m_c = gen_meas_circuits(transpiled_cir, ['zzz'])
        res = inst.execute(m_c)
        counts = res.get_counts()
        
        #-----#
        # One qubit task
        #-----#
        ansatz = anz.AnsatzFromFunction(anz._1qubit_2_params_XZ)
        X_SOL = np.pi/4 * np.ones(2)
            
        coeffs = np.array([0.5, -0.5, np.sqrt(1/2)])
        cost1q = OneQXYZ(ansatz, inst, coeffs = coeffs, decompose = False)
        assert np.abs(cost1q(X_SOL) - 1) < 0.01, "For this ansatz, parameters, cost function should be close to one (stat fluctuations)"

        cost1q = OneQXYZ(ansatz, inst, coeffs = coeffs, decompose = True)
        assert np.all(np.abs(2*cost1q(X_SOL) - 1 -coeffs) < 0.05)  
        
        #-----#
        # GHZ
        #-----#
        # Create an ansatz capable of generating a GHZ state (not the most obvious 
        # one here) with the set of params X_SOL
        X_SOL = np.pi/2 * np.array([1.,1.,2.,1.,1.,1.])
        X_LOC = np.pi/2 * np.array([1., 0., 4., 0., 3., 0.])
        X_RDM = np.random.uniform(0.0, 2*pi, size=(6,1))
                
        # Verif the values of the different GHZ cost
        # Fidelity
        ansatz = anz.AnsatzFromFunction(anz._GHZ_3qubits_6_params_cx0)
        ghz_cost = GHZPauliCost(ansatz=ansatz, instance = inst)
        assert ghz_cost(X_SOL) == 1.0, "For this ansatz, parameters, cost function should be one"
        assert np.abs(ghz_cost(X_LOC) - 0.5) < 0.1, "For this ansatz and parameters, the cost function should be close to 0.5 (up to sampling error)"
        
        test_batch = ghz_cost([X_SOL] * 11)
        
        # Witnesses inspired cost functions: they are different compared to the fidelity
        # but get maximized only when the state is the right one
        ghz_witness1 = GHZWitness1Cost(ansatz=ansatz, instance = inst, N=3, nb_params=6)
        assert ghz_witness1(X_SOL) == 1.0, "For this ansatz, parameters, cost function should be one"
        assert np.abs(ghz_witness1(X_LOC) - 0.31) < 0.1, "For this ansatz and parameters, the cost function should be close to 0.31 (up to sampling error)"

        ghz_witness2 = GHZWitness2Cost(ansatz=ansatz, instance = inst, N=3, nb_params=6)
        assert ghz_witness2(X_SOL) == 1.0, "For this ansatz, parameters, cost function should be one"
        assert np.abs(ghz_witness2(X_LOC) + 0.5) < 0.1, "For this ansatz and parameters, the cost function should be close to 0.31 (up to sampling error)"    
        
        
        
        #-----#
        # Cyclical graph states
        #-----#

        ansatz = anz.AnsatzFromFunction(anz._GraphCycl_6qubits_6params)
        X_SOL = np.pi/2 * np.ones(ansatz.nb_params) # sol of the cycl graph state for this ansatz
        X_RDM = np.array([1.70386471,1.38266762,3.4257722,5.78064,3.84102323,2.37653078])
        #X_RDM = np.random.uniform(low=0., high=2*np.pi, size=(N_params,))
        

        graph_cost = GraphCyclPauliCost(ansatz=ansatz, instance = inst)
        
        fid_opt = graph_cost(X_SOL)
        fid_rdm = graph_cost(X_RDM)
        assert fid_opt == 1.0, "For this ansatz, parameters, cost function should be one"
        assert (fid_opt-fid_rdm) > 1e-4, "For this ansatz, parameters, cost function should be one"
        
        if False: # don't test the broken witness (raises warning now)
            graph_cost1 = GraphCyclWitness1Cost(ansatz=ansatz, instance = inst)
            cost1_opt = graph_cost1(X_SOL)
            cost1_rdm = graph_cost1(X_RDM)
            assert cost1_opt == 1.0, "For this ansatz, parameters, cost function should be one"
            assert  (fid_rdm - cost1_rdm) > 1e-4, "cost function1 should be lower than true fid"
        
        graph_cost2 = GraphCyclWitness2Cost(ansatz=ansatz, instance = inst)
        cost2_opt = graph_cost2(X_SOL)
        cost2_rdm = graph_cost2(X_RDM)
        assert cost2_opt == 1.0, "For this ansatz, parameters, cost function should be one"
        assert  (fid_rdm - cost2_rdm) > 1e-4, "cost function should be lower than true fid"
        
        graph_cost2full = GraphCyclWitness2FullCost(ansatz=ansatz, instance = inst)
        cost2full_opt = graph_cost2full(X_SOL)
        cost2full_rdm = graph_cost2full(X_RDM)
        assert cost2full_opt == 1.0, "For this ansatz, parameters, cost function should be one"
        assert  (fid_rdm - cost2_rdm) > 1e-4, "cost function should be lower than true fid"
        assert  np.abs(cost2full_rdm - cost2_rdm) < 0.1, "both cost function should be closed"
        
        X_SOL = np.pi/2 * np.array([1.,1.,2.,1.,1.,1.])
        circs0 = ghz_cost.meas_circuits
        circs1 = ghz_witness2.meas_circuits
        circs0 = bind_params(circs0, X_SOL, ghz_cost.qk_vars)
        circs1 = bind_params(circs1, X_SOL, ghz_witness2.qk_vars)
        circs = circs0 + circs1
        res = inst.execute(circs, had_transpiled=True)
        
        assert ghz_cost.evaluate_cost(res) == 1.0, "For passing in results object, check the solutions are correct"
        assert ghz_witness2.evaluate_cost(res) == 1.0, "For passing in results object, check the solutions are correct"

        
    # Basic checks for random XY cost function (appears to be working fine)
    h_field = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    h_xy = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    ansatz = anz.AnsatzFromFunction(anz._GHZ_3qubits_6_params_cx0)
    xy_cost = RandomXYCost(ansatz, inst, h_field)
    circs = xy_cost.bind_params_to_meas([0,0,0,0,0,0])
    res = inst.execute(circs)
    assert xy_cost.evaluate_cost(res) == -3.0, "Field only Hamiltonian in logical 0 == -1 z-projection"
    assert xy_cost([0,0,0,pi,pi,pi]) == 3.0, "Field only Hamiltonian in logical 0 == -1 z-projection"

    xy_cost = RandomXYCost(ansatz, inst, h_xy)
    assert abs(xy_cost([0,0,0,0,0,0])) < 0.08, "z = -1 state should be close to zero (this may fail very randomly)" 
    assert abs(xy_cost([0,0,0,pi/2,pi/2,pi/2]) - 1) < 0.08, "XY product state state should be close to 1 (this may fail very randomly)" 