"""
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
    num_rand : int
        Number of random unitaries to use
    seed : int, optional
        Random number seed for reproducibility

    Returns
    -------
    purity_circuits : list of qiskit circuits
        Copies of input circuit(s), with random unitaries added to the end
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
    submitting copies of the same circuits.
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
        """ """
        return self._circ_name

    @circ_name.setter
    def circ_name(self, circ_name):
        """ """
        self._circ_name = circ_name
        for idx, circ in enumerate(self._meas_circuits):
            circ.name = self._circ_name(idx)

    def circuits(self, point):
        """
        Yield circuits
        """
        if not isinstance(point, np.ndarray):
            raise TypeError("point passed to RandomMeasurementHandler as type "
                            + f'{type(point)}')

        if (
            self._last_point is not None
            and np.all(np.isclose(self._last_point, point))
        ):
            return []

        self._last_point = point
        return bind_params(self._meas_circuits, point, self.ansatz.params)

    def reset(self):
        """
        Lose memory of having yielded circuits
        """
        self._last_point = None
