#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Apr 20 13:15:21 2020

@author: Chris, Kiran, Fred
basic script to define ansatz classes to streamline cost function creation

Anything that conforms to the interface should be able to be passed into a 
cost function

CHANGES: I've rename num_qubit to nb_qubit for consistency with everyhitng else
TODO: Change params -> qk_vars to match cost interface better?
"""
# ===================
# Define ansatz and initialize costfunction
# ===================

# list of * contents
__all__ = [
    # ansatz classes
    'AnsatzInterface',
    'BaseAnsatz',
    'TrivialAnsatz',
    'AnsatzFromFunction',
    'RandomAnsatz',
    'RegularXYZAnsatz',
    'RegularU3Ansatz',
    # helper functions
    'count_params_from_func',

]

import abc
import pdb
import sys
import random

import qiskit as qk
import numpy as np

class AnsatzInterface(metaclass=abc.ABCMeta):
    """Interface for a parameterised ansatz. Specifies an object that must have five
    properties: a depth, a list of qiskit parameters objects, a circuit, the number 
    of qubits and the number of parameters.
    """
    @property
    @abc.abstractmethod
    def depth(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def params(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def circuit(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nb_qubits(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nb_params(self):
        raise NotImplementedError

class BaseAnsatz(AnsatzInterface):
    """ """

    def __init__(
                 self,
                 num_qubits,
                 depth,
                 **kwargs
                ):
        self._nb_qubits = num_qubits
        self._depth = depth

        # make circuit and generate parameters
        self._params = self._generate_params()
        self._nb_params = len(self._params)
        self._circuit = self._generate_circuit()

    def _generate_params(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError

    def _generate_circuit(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError

    @property
    def depth(self):
        return self._depth

    @property
    def circuit(self):
        return self._circuit

    @property
    def params(self):
        return self._params

    @property
    def nb_qubits(self):
        return self._nb_qubits

    @property
    def nb_params(self):
        return self._nb_params

class TrivialAnsatz(BaseAnsatz):
    """ Ansatz that wraps a fixed circuit """

    def __init__(self,fixed_circuit):
        """ """
        self._generate_circuit = (lambda: fixed_circuit)
        self._generate_params = (lambda: [])

        # explicitly call base class initialiser
        super(TrivialAnsatz,self).__init__(fixed_circuit.num_qubits,0)

class AnsatzFromFunction(AnsatzInterface):
    """ Returns an instance of the GHZ parameterized class"""
    def __init__(self, ansatz_function, x_sol = None):
        self.x_sol = x_sol
        self._nb_params = count_params_from_func(ansatz_function)
        self._params = self._generate_params()
        self._circuit = self._generate_circuit(ansatz_function)
        self._nb_qubits = self._circuit.num_qubits
        self._depth = self._circuit.depth()

    def _generate_params(self):
        """ Generate qiskit variables to be bound to a circuit previously was
            in Cost class"""
        name_params = ['R'+str(i) for i in range(self._nb_params)]
        params = [qk.circuit.Parameter(n) for n in name_params]
        return params

    def _generate_circuit(self, ansatz_function):
        """ To be implemented in the subclasses """
        return ansatz_function(self._params)
    
    def _reorder_params(self):
        """ Probably useless, but ensures ordred list of params (careful when loading from pkl files)"""
        names = [p.name.split('R')[1] for p in self._params]
        di = dict(zip(names, self._params))
        reordered = []
        for ii in range(self.nb_params):
            reordered.append(di[str(ii)])
        self._params = reordered

    @property
    def depth(self):
        return self._depth

    @property
    def circuit(self):
        return self._circuit

    @property
    def params(self):
        return self._params

    @property
    def nb_qubits(self):
        return self._nb_qubits

    @property
    def nb_params(self):
        return self._nb_params

class RandomAnsatz(BaseAnsatz):
    """ """

    def __init__(self,*args,
                 seed=None,
                 gate2='CX',
                ):

        # set random seed if passed
        if seed is not None:
            random.seed(seed)

        # set two-qubit gate
        self.gate2 = gate2

        # explicitly call base class initialiser
        super(RandomAnsatz,self).__init__(*args)

    def _generate_params(self):
        """ """
        nb_params = 2*(self._nb_qubits-1)*self._depth + self._nb_qubits
        name_params = ['R'+str(i) for i in range(nb_params)]
        return [qk.circuit.Parameter(n) for n in name_params]

    def _generate_circuit(self):
        """ """

        # the set number of entangling pairs to distribute randomly
        ent_pairs = [(i, i + 1) for i in range(self._nb_qubits - 1) for _ in range(self._depth)]
        random.shuffle(ent_pairs)
        
        # keep track of where not to apply a entangling gate again
        just_entangled = set()
            
        # keep track of where its worth putting a parameterised gate
        needs_rgate = [True] * self._nb_qubits

        # make circuit obj and list of parameter obj's created
        qc = qk.QuantumCircuit(self._nb_qubits)

        # parse entangling gate arg
        if self.gate2=='CZ':
            ent_gate = qc.cz
        elif self.gate2=='CX':
            ent_gate = qc.cx
        else:
            print("entangling gate not recognised, please specify: 'CX' or 'CZ'", file=sys.stderr)
            raise ValueError
            
        # array of single qubit e^{-i\theta/2 \sigma_i} type gates, we will
        # randomly draw from these
        single_qubit_gates = [qc.rx,qc.ry,qc.rz]
        
        # track next parameter to use
        param_counter = 0

        # consume list of pairs to entangle
        while ent_pairs:
            for i in range(self._nb_qubits):
                if needs_rgate[i]:
                    (single_qubit_gates[random.randint(0,2)])(self._params[param_counter],i)
                    param_counter += 1
                    needs_rgate[i] = False
                    
            for k, pair in enumerate(ent_pairs):
                if pair not in just_entangled:
                    break
            i, j = ent_pairs.pop(k)
            ent_gate(i, j)
            
            just_entangled.add((i, j))
            just_entangled.discard((i - 1, j - 1))
            just_entangled.discard((i + 1, j + 1))
            needs_rgate[i] = needs_rgate[j] = True
        
        for i in range(self._nb_qubits):
            if needs_rgate[i]:
                (single_qubit_gates[random.randint(0,2)])(self._params[param_counter],i)
                param_counter += 1
        
        return qc

class RegularXYZAnsatz(BaseAnsatz):
    """ """

    def _generate_params(self):
        """ """
        nb_params = self._nb_qubits*(self._depth+1)
        name_params = ['R'+str(i) for i in range(nb_params)]
        return [qk.circuit.Parameter(n) for n in name_params]

    def _generate_circuit(self):
        """ """

        N = self._nb_qubits
        barriers = True
        
        qc = qk.QuantumCircuit(N)
        
        egate = qc.cx # entangle with CNOTs
        single_qubit_gate_sequence = [qc.rx,qc.ry,qc.rz] # eisert scheme alternates RX, RY, RZ 
        
        # initial round in the Eisert scheme is fixed RY rotations at 
        # angle pi/4
        qc.ry(np.pi/4,range(N))
        l,r = 2*np.arange(N//2),2*np.arange(N//2)+1
        if len(l)==1:
            egate(l[0],r[0])
        elif len(l)>1:
            egate(l,r)
        l,r = 2*np.arange(N//2-1+(N%2))+1,2*np.arange(N//2-1+(N%2))+2
        if len(l)==1:
            egate(l[0],r[0])
        elif len(l)>1:
            egate(l,r)
        if barriers:
            qc.barrier()
        
        param_counter = 0
        for r in range(self._depth):

            # add parameterised single qubit rotations
            for q in range(N):
                gate = single_qubit_gate_sequence[r % len(single_qubit_gate_sequence)]
                gate(self._params[param_counter],q)
                param_counter += 1

            # add entangling gates
            l,r = 2*np.arange(N//2),2*np.arange(N//2)+1
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            l,r = 2*np.arange(N//2-1+(N%2))+1,2*np.arange(N//2-1+(N%2))+2
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            if barriers:
                qc.barrier()
        
        # add final round of parameterised single qubit rotations
        for q in range(N):
            gate = single_qubit_gate_sequence[self._depth % len(single_qubit_gate_sequence)]
            gate(self._params[param_counter],q)
            param_counter += 1

        return qc

class RegularU3Ansatz(BaseAnsatz):
    """ """

    def _generate_params(self):
        """ """
        nb_params = self._nb_qubits*(self._depth+1)*3
        name_params = ['R'+str(i) for i in range(nb_params)]
        return [qk.circuit.Parameter(n) for n in name_params]

    def _generate_circuit(self):
        """ """

        N = self._nb_qubits
        barriers = True
        
        qc = qk.QuantumCircuit(N)

        egate = qc.cx # entangle with CNOTs

        param_counter = 0
        for r in range(self._depth):

            # add parameterised single qubit rotations
            for q in range(N):                  
                qc.u3(*[self._params[param_counter+i] for i in range(3)],q)
                param_counter += 3

            # add entangling gates
            l,r = 2*np.arange(N//2),2*np.arange(N//2)+1
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            l,r = 2*np.arange(N//2-1+(N%2))+1,2*np.arange(N//2-1+(N%2))+2
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            if barriers:
                qc.barrier()

        # add final round of parameterised single qubit rotations
        for q in range(N):
            qc.u3(*[self._params[param_counter+i] for i in range(3)],q)
            param_counter += 3
        
        return qc


class RegularU2Ansatz(BaseAnsatz):
    """ """

    def _generate_params(self):
        """ """
        nb_params = self._nb_qubits*(self._depth+1)*2
        name_params = ['R'+str(i) for i in range(nb_params)]
        return [qk.circuit.Parameter(n) for n in name_params]

    def _generate_circuit(self):
        """ """

        N = self._nb_qubits
        barriers = True
        
        qc = qk.QuantumCircuit(N)

        egate = qc.cx # entangle with CNOTs

        param_counter = 0
        for r in range(self._depth):

            # add parameterised single qubit rotations
            for q in range(N):
                qc.u2(*[self._params[param_counter+i] for i in range(2)],q)
                param_counter += 2

            # add entangling gates
            l,r = 2*np.arange(N//2),2*np.arange(N//2)+1
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            l,r = 2*np.arange(N//2-1+(N%2))+1,2*np.arange(N//2-1+(N%2))+2
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            if barriers:
                qc.barrier()

        # add final round of parameterised single qubit rotations
        for q in range(N):
            qc.u2(*[self._params[param_counter+i] for i in range(2)],q)
            param_counter += 2
        
        return qc


class RegularRandomU3ParamAnsatz(BaseAnsatz):
    """ 
    Regular entangeling gates + random U3 unitaries per layer. Only 1 parameter per u3 unitary
    is used. 

        
    """
    def __init__(self,*args,seed=None):
        if seed is not None:
            np.random.seed(seed)
        super().__init__(*args)

    def _generate_params(self):
        """ """
        nb_params = self._nb_qubits*(self._depth+1)*1
        name_params = ['R'+str(i) for i in range(nb_params)]
        return [qk.circuit.Parameter(n) for n in name_params]

    def _generate_circuit(self):
        """ """

        N = self._nb_qubits
        barriers = True
        
        qc = qk.QuantumCircuit(N)

        egate = qc.cx # entangle with CNOTs

        param_counter = 0
        for r in range(self._depth):

            # add parameterised single qubit rotations
            for q in range(N):
                choice = 3*np.random.rand()
                if choice <= 1:
                    rot = [self._params[param_counter], 0, 0]
                elif choice >= 2:
                    rot = [0, self._params[param_counter], 0]
                else:
                    rot = [0, 0, self._params[param_counter]]
                qc.u3(*rot,q)
                param_counter += 1

            # add entangling gates
            l,r = 2*np.arange(N//2),2*np.arange(N//2)+1
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            l,r = 2*np.arange(N//2-1+(N%2))+1,2*np.arange(N//2-1+(N%2))+2
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            if barriers:
                qc.barrier()

        # add final round of parameterised single qubit rotations
        for qu in range(N):
            choice = 3*np.random.rand()
            if choice <= 1:
                rot = [self._params[param_counter], 0, 0]
            elif choice >= 2:
                rot = [0, self._params[param_counter], 0]
            else:
                rot = [0, 0, self._params[param_counter]]
            qc.u3(*rot,qu)
            param_counter += 1
        
        return qc

# ------------------------------------------------------------------------------
def count_params_from_func(ansatz):
    """ Counts the number of parameters that the ansatz function accepts"""
    call_list = [0]
    for ii in range(100):
        try:
            ansatz(call_list)
            return len(call_list)
        except IndexError:
            call_list += [0]
    
# ----------------------------------------
# Useful function circuits that have been checked
# ----------------------------------------
def _1qubit_2_params_XZ(params, barriers = True):
    """ Returns function handle for 1 qubit Rx Rz preparation
    With barrier by default"""
    logical_qubits = qk.QuantumRegister(1, 'logical')
    c = qk.QuantumCircuit(logical_qubits)
    c.rx(params[0], qubit=0)
    if barriers: c.barrier()
    c.rz(params[1], qubit=0)
    if barriers: c.barrier()
    return c

def _GHZ_3qubits_6_params_cx0(params, barriers = False):
    """ Returns function handle for 6 param ghz state"""
    logical_qubits = qk.QuantumRegister(3, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.rx(params[0], 0)
    c.rx(params[1], 1)
    c.ry(params[2], 2)
    if barriers: c.barrier()
    c.cnot(0,2) 
    c.cnot(1,2) 
    if barriers: c.barrier()
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    if barriers: c.barrier()
    return c

def _GHZ_3qubits_6_params_cx1(params, barriers = False):
    """ Returns function handle for 6 param ghz state 1 swap"""
    logical_qubits = qk.QuantumRegister(3, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.ry(params[2], 0)
    c.rx(params[1], 1)
    c.rx(params[0], 2)
    c.swap(0, 2)
    if barriers: c.barrier()
    c.cnot(0,2) 
    c.cnot(1,2) 
    if barriers: c.barrier()
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    if barriers: c.barrier()
    return c

def _GHZ_3qubits_6_params_cx2(params, barriers = False):
    """ Returns function handle for 6 param ghz state 2 swaps"""
    logical_qubits = qk.QuantumRegister(3, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.rx(params[1], 0) 
    c.ry(params[2], 1) 
    c.rx(params[0], 2) 
    c.swap(0, 2)
    c.swap(1, 2)
    if barriers: c.barrier()
    c.cnot(0,2) 
    c.cnot(1,2) 
    if barriers: c.barrier()
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    if barriers: c.barrier()
    return c

def _GHZ_3qubits_6_params_cx3(params, barriers = False):
    """ Returns function handle for 6 param ghz state 3 swaps"""
    logical_qubits = qk.QuantumRegister(3, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.rx(params[1], 0) 
    c.rx(params[0], 1) 
    c.ry(params[2], 2) 
    c.swap(0, 2)
    c.swap(1, 2)
    c.swap(0, 2)
    if barriers: c.barrier()
    c.cnot(0,2) 
    c.cnot(1,2) 
    if barriers: c.barrier()
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    if barriers: c.barrier()
    return c

def _GHZ_3qubits_6_params_cx4(params, barriers = False):
    """ Returns function handle for 6 param ghz state 4 swaps"""
    logical_qubits = qk.QuantumRegister(3, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.ry(params[2], 0) 
    c.rx(params[0], 1) 
    c.rx(params[1], 2) 
    c.swap(0, 2)
    c.swap(1, 2)
    c.swap(0, 2)
    c.swap(1, 2)
    if barriers: c.barrier()
    c.cnot(0,2) 
    c.cnot(1,2) 
    if barriers: c.barrier()
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    if barriers: c.barrier()
    return c
    
def _GraphCycl_6qubits_6params(params, barriers = False):        
    """ Returns handle to cyc6 cluster state with c-phase gates"""
    logical_qubits = qk.QuantumRegister(6, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.ry(params[0],0)
    c.ry(params[1],1)
    c.ry(params[2],2)
    c.ry(params[3],3)
    c.ry(params[4],4)
    c.ry(params[5],5)
    if barriers: c.barrier()
    c.cz(0,1)
    c.cz(2,3)
    c.cz(4,5)
    if barriers: c.barrier()
    c.cz(1,2)
    c.cz(3,4)
    c.cz(5,0)
    if barriers: c.barrier()
    return c

def _GraphCycl_6qubits_6params_inefficient(params, barriers = False):        
    """ Returns handle to cyc6 cluster state with cry() gates"""
    logical_qubits = qk.QuantumRegister(6, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.ry(params[0],0)
    c.ry(params[1],1)
    c.ry(params[2],2)
    c.ry(params[3],3)
    c.ry(params[4],4)
    c.ry(params[5],5)
    if barriers: c.barrier()
    c.crz(np.pi, 0,1)
    c.crz(np.pi, 2,3)
    c.crz(np.pi, 4,5)
    if barriers: c.barrier()
    c.crz(np.pi, 1,2)
    c.crz(np.pi, 3,4)
    c.crz(np.pi, 5,0)
    if barriers: c.barrier()
    return c

def _GraphCycl_6qubits_24params(params, barriers = False):
    """ Ansatz to be refined, too many params - BO doens't converge"""
    logical_qubits = qk.QuantumRegister(6, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.h(0)
    c.h(1)
    c.h(2)
    c.h(3)
    c.h(4)
    c.h(5)
    if barriers: c.barrier()
    c.ry(params[0], 0)
    c.ry(params[1], 1)
    c.ry(params[2], 2)
    c.ry(params[3], 3)
    c.ry(params[4], 4)
    c.ry(params[5], 5)
    if barriers: c.barrier()
    c.cnot(0,1) 
    c.cnot(2,3) 
    c.cnot(4,5)
    if barriers: c.barrier()
    c.rz(params[6], 0)
    c.rz(params[7], 1)
    c.rz(params[8], 2)
    c.rz(params[9], 3)
    c.rz(params[10], 4)
    c.rz(params[11], 5)
    if barriers: c.barrier()
    c.cnot(1,2) 
    c.cnot(3,4)
    c.cnot(5,0)
    if barriers: c.barrier()
    c.u2(params[12], params[13], 0)
    c.u2(params[14], params[15], 1)
    c.u2(params[16], params[17], 2)
    c.u2(params[18], params[19], 3)
    c.u2(params[20], params[21], 4)
    c.u2(params[22], params[23], 5)
    if barriers: c.barrier()
    return c

