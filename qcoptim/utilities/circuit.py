"""
Circuit utilities
"""

import numpy as np
from numpy import pi, sin, cos
from numpy import arccos as acos
from numpy import arctan as atan

from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.utils import QuantumInstance
from qiskit.quantum_info import random_unitary

from .core import prefix_to_names
from .pytket import compile_for_backend


def simplify_rotation_angles(circuit):
    """
    Transpiled circuits can sometimes end up with rotation gates where the
    angles are given as complicated expressions (even when the parameters are
    bound), which often cannot be drawn on screen. This function calls `eval`
    on these expression and generates a copy of the circuit containing only
    float valued angles.

    Assumes circuit has been transpiled into the qiskit basis set:
    {sqrt{X}, Rz, CX} and so only Rz gates are rotations.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Circuit to simplify

    Returns
    -------
    simplified_circuit : qiskit.QuantumCircuit
        Copy of the circuit with rotation angles evaluated
    """
    qasm_string = circuit.qasm()

    eval_lines = []
    for line in qasm_string.split('\n'):
        if line[:2] == 'rz':
            for count_back in range(len(line)):
                if line[-count_back] == ')':
                    break
            eval_str = line[3:-count_back]
            val = eval(eval_str)
            eval_lines.append(line[:3] + f'{val}' + line[-count_back:])
        else:
            eval_lines.append(line)
    qasm_eval = '\n'.join(eval_lines)

    return QuantumCircuit.from_qasm_str(qasm_eval)


def transpile_circuit(
    circuit,
    instance,
    method,
    enforce_bijection=False,
    **transpile_args,
):
    """
    Transpile the circuit for a backend passed as a quantum instance, using
    method specified by method arg.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Circuit to transpile
    instance : qiskit.utils.QuantumInstance obj
        Instance to use as reference for transpiling
    method : str, optional
        Method to use for transpiling, supported options:
            -> "instance" : use quantum instance
            -> "pytket" : use pytket, targeting instance's backend
    enforce_bijection : boolean, optional
        If set to True, will raise ValueError if the transpiler map found
        is not a bijection
    **transpile_args : dict
        Other args to pass to the transpiler

    Returns
    -------
    qiskit.QuantumCircuit
        The transpiled circuit
    """

    # add measurements, these will be used to infer transpiler_map
    tmp = circuit.copy()
    tmp.measure_all()

    # run transpiler with method
    if method == 'instance':
        t_circ = instance.transpile(tmp)[0]
    elif method == 'pytket':
        t_circ = compile_for_backend(instance.backend, tmp, **transpile_args)
    else:
        raise ValueError('Transpiler method: '+f'{method}'+', not recognized.')

    # extract transpiler map from circuit data
    transpiler_map = {}
    for instruction in t_circ.data:
        if isinstance(instruction[0], Measure):
            qubit = instruction[1][0].index
            clbit = instruction[2][0].index
            if (
                enforce_bijection
                and (clbit in transpiler_map.keys()
                     or qubit in transpiler_map.values())
            ):
                raise ValueError(
                    'Transpiler map is not a bijection. Possibly ansatz'
                    + ' circuit already contained measurements,'
                    + ' or transpilation reused qubits for final'
                    + ' measurements.'
                )
            transpiler_map[clbit] = qubit

    # remove final measurements
    t_circ.remove_final_measurements()

    return t_circ, transpiler_map


def bind_params(circ, param_values, param_variables, param_name=None):
    """
    Take a list of circuits with bindable parameters and bind the values
    passed according to the param_variables Returns the list of circuits with
    bound values DOES NOT MODIFY INPUT (i.e. hardware details??)

    As of (at least) qiskit 0.25.2 circuits are renamed (with a suffix) when
    parameters are bound. This breaks some of our code so we remove that suffix
    here.

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

    # bind circuits but preserve names
    bound_circ = []
    val_dict = dict(zip(param_variables, param_values))
    for cc in circ:
        circ_name = cc.name
        tmp = cc.bind_parameters(val_dict)
        tmp.name = circ_name
        bound_circ.append(tmp)

    # (optionally) add prefix to circuit names
    if param_name is not None:
        bound_circ = prefix_to_names(bound_circ, param_name)

    return bound_circ


def add_random_measurements(circuit, num_rand, active_qubits=None, seed=None):
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
    active_qubits : list-like iterable, optional
        If passed, random measurements will only be applied to these qubits.
    seed : int, optional
        Random number seed for reproducibility

    Returns
    -------
    purity_circuits : list of qiskit circuits
        Copies of input circuit, with random unitaries added to the end
    """
    rand_state = np.random.default_rng(seed)

    # by default apply to all qubits
    if active_qubits is None:
        active_qubits = list(range(circuit.num_qubits))

    rand_meas_circuits = []
    for _ in range(num_rand):

        # copy circuit to preserve registers, but remove any final measurements
        new_circ = circuit.copy()
        new_circ.remove_final_measurements()

        # add random single qubit unitaries
        for qb_idx in active_qubits:
            rand_gate = random_unitary(2, seed=rand_state)
            new_circ.append(rand_gate, [qb_idx])

        # adapted from qiskit.QuantumCircuit.measure_active() source
        qubits_to_measure = [
            qubit for qubit in new_circ.qubits if qubit.index in active_qubits
        ]
        new_creg = new_circ._create_creg(len(active_qubits), 'meas')
        new_circ.add_register(new_creg)
        new_circ.barrier()
        new_circ.measure(qubits_to_measure, new_creg)

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
        transpiler='instance',
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
        transpiler : str, optional
            Choose how to transpile circuits, current options are:
                'instance' : use quantum instance
                'pytket' : use pytket compiler
        """
        self.ansatz = ansatz
        self.instance = instance
        self.num_random = num_random
        self.seed = seed
        if circ_name is None:
            def circ_name(idx):
                return 'HaarRandom' + f'{idx}'
        self._circ_name = circ_name

        # transpile ansatz circuit
        t_ansatz_circ = self.ansatz.transpiled_circuit(
            self.instance, method=transpiler, enforce_bijection=True)

        # make and name measurement circuits
        self._meas_circuits = add_random_measurements(
            t_ansatz_circ,
            self.num_random,
            active_qubits=self.ansatz.transpiler_map.values(),
            seed=seed,
        )

        # convert measurement instructions to the backend's gateset
        simple_instance = QuantumInstance(
            self.instance.backend, optimization_level=0)
        self._meas_circuits = simple_instance.transpile(self._meas_circuits)

        # name circuits
        for idx, circ in enumerate(self._meas_circuits):
            circ.name = self._circ_name(idx)

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
