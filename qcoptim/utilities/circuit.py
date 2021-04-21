"""
Circuit utilities
"""

import numpy as np

from qiskit.quantum_info import random_unitary

from .core import prefix_to_names


def bind_params(circ, param_values, param_variables, param_name=None):
    """
    Take a list of circuits with bindable parameters and bind the values
    passed according to the param_variables Returns the list of circuits with
    bound values DOES NOT MODIFY INPUT (i.e. hardware details??)

    Parameters
    ----------
    circ : qiskit circtuit(s)
        Single or list of quantum circuits with the same qk_vars
    param_values : a 1d array of parameters (i.e. correspond to a single
        set of parameters)
    param_variables : list of qk_vars, it should match element-wise
        to the param_values
    param_name : str if not None it will used to prepend the names
        of the circuits created

    Returns
    -------
    bound quantum circuits
    """
    if not isinstance(circ, list):
        circ = [circ]

    val_dict = dict(zip(param_variables, param_values))
    bound_circ = [cc.bind_parameters(val_dict) for cc in circ]
    if param_name is not None:
        bound_circ = prefix_to_names(bound_circ, param_name)
    return bound_circ


def add_random_measurements(circuit, num_rand, seed=None):
    """
    Add single qubit measurements in Haar random basis to all the qubits, at
    the end of the circuit. Copies the circuit so preserves registers,
    parameters and circuit name. Independent of what measurements were in the
    input circuit, all qubits will be measured.

    Parameters
    ----------
    circuit : qiskit circuit
        Circuit to add random measurements to
    num_rand : int
        Number of random unitaries to use
    seed : int, optional
        Random number seed for reproducibility

    Returns
    -------
    purity_circuits : list of qiskit circuits
        Copies of input circuit, with random unitaries added to the end
    """
    rand_state = np.random.default_rng(seed)

    rand_meas_circuits = []
    for _ in range(num_rand):

        # copy circuit to preserve registers, but remove any final measurements
        new_circ = circuit.copy()
        new_circ.remove_final_measurements()
        # add random single qubit unitaries
        for qb_idx in range(new_circ.num_qubits):
            rand_gate = random_unitary(2, seed=rand_state)
            new_circ.append(rand_gate, [qb_idx])
        new_circ.measure_all()
        rand_meas_circuits.append(new_circ)

    return rand_meas_circuits


class RandomMeasurementHandler():
    """
    Several tasks (e.g. measuring cross-fidelity, purity boosting error
    mitigation) require measuring all of the active qubits in Haar random
    basis. If we want to do some of these things simulataneously we risk
    submitting copies of the same circuits. This class encapsulates the task of
    generating the circuits needed and locks on consecutive requests for the
    same set of circuits, making it possible to avoid this problem.
    """
    def __init__(
        self,
        ansatz,
        instance,
        num_random,
        seed=None,
        circ_name=None,
    ):
        """
        Parameters
        ----------
        ansatz : class implementing ansatz interface
            Ansatz obj
        instance : qiskit.aqua.QuantumInstance
            Quantum instance to use
        num_random : int
            Number of random basis to generate
        seed : int, optional
            Seed for generating random basis
        circ_name : callable, optional
            Function used to name circuits, should have signature `int -> str`
            and preferably should prefix the int, e.g. return something like
            'some-str'+str(int)`
        """
        self.ansatz = ansatz
        self.instance = instance
        self.num_random = num_random
        self.seed = seed
        if circ_name is None:
            def circ_name(idx):
                return 'HaarRandom' + f'{idx}'
        self._circ_name = circ_name

        # make, name and transpile circuits
        self._meas_circuits = add_random_measurements(self.ansatz.circuit,
                                                      self.num_random,
                                                      seed=seed)
        for idx, circ in enumerate(self._meas_circuits):
            circ.name = self._circ_name(idx)
        self._meas_circuits = self.instance.transpile(self._meas_circuits)

        # used to allow shared use without generating redundant circuit copies
        self._last_point = None

    @property
    def circ_name(self):
        """
        Returns
        -------
        callable
            Obj's circuit naming function, maps `int -> str`
        """
        return self._circ_name

    @circ_name.setter
    def circ_name(self, circ_name):
        """
        Setter for circuit naming function. If the function is changed during
        use we want to rename all of the obj's stored circuits.

        Parameters
        ----------
        circ_name : callable
            New function for circuit naming should  act as `int -> str`
        """
        self._circ_name = circ_name
        for idx, circ in enumerate(self._meas_circuits):
            circ.name = self._circ_name(idx)

    def circuits(self, evaluate_at):
        """
        Yield circuits, if multiple consecutive requests are made for the
        circuits at the same point only the first call will return the
        circuits, after that [] is returned each time. This lock releases if a
        new point(s) is requested.

        Parameters
        ----------
        evaluate_at : numpy.ndarray
            Point (1d) or points (2d) to bind circuits at

        Returns
        -------
        qiskit.QuantumCircuit
            Transpiled quantum circuits
        """
        # special case, circuit has no parameters to bind
        if self.ansatz.nb_params == 0:
            if self._last_point is None:
                self._last_point = 0
                return self._meas_circuits
            return []

        if not isinstance(evaluate_at, np.ndarray):
            raise TypeError("evaluate_at passed has type "
                            + f'{type(evaluate_at)}')

        if evaluate_at.ndim > 2:
            raise ValueError('evaluate_at has too many dimensions.')

        if (
            self._last_point is not None
            and np.all(np.isclose(self._last_point, evaluate_at))
        ):
            return []

        self._last_point = evaluate_at.copy()

        if evaluate_at.ndim == 2:
            circs = []
            for point in evaluate_at:
                circs += bind_params(self._meas_circuits, point,
                                     self.ansatz.params)
            return circs

        return bind_params(self._meas_circuits, evaluate_at,
                           self.ansatz.params)

    def reset(self):
        """
        Lose memory of having yielded circuits
        """
        self._last_point = None
