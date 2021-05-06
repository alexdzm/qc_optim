"""
"""

import sys

import numpy as np
import quimb as qu

from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.circuit import Measure, Qubit, Clbit
from qiskit.result import Result
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.utils.backend_utils import is_ibmq_provider
# these Classes are unfortunately now DEPRECATED by qiskit
from qiskit.aqua.operators import WeightedPauliOperator as wpo
from qiskit.aqua.operators import TPBGroupedWeightedPauliOperator as groupedwpo

from ..utilities import qTNfromQASM

from .core import CostInterface


def _map_qargs(input_qargs, transpiler_map, transpiled_register):
    """
    """
    mapped_qargs = []
    for in_qubit in input_qargs:
        mapped_qargs.append(
            Qubit(transpiled_register, transpiler_map[in_qubit.index]))
    return mapped_qargs


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
        group=True,
        transpiler='instance',
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
        group : boolean, optional
            Set to False to turn off operator grouping
        transpiler : str, optional
            Choose how to transpile circuits, current options are:
                'instance' : use quantum instance
                'pytket' : use pytket compiler
        """
        self.last_evaluation = None
        self.ansatz = ansatz
        self.instance = instance

        # check type of passed operators
        if not isinstance(weighted_pauli_operators, wpo):
            raise TypeError

        # ensure the ansatz and qubit Hamiltonians have same number of qubits
        if not weighted_pauli_operators.num_qubits == self.ansatz.nb_qubits:
            raise ValueError('Number of qubits in WeightedPauliOperator'
                             + ' different from ansatz circuit.')

        # currently using `unsorted_grouping` method, which is a greedy method.
        # Sorting method could be controlled with a kwarg
        if group:
            self._weighted_operators = groupedwpo.unsorted_grouping(
                weighted_pauli_operators)
        else:
            self._weighted_operators = weighted_pauli_operators

        # transpile ansatz circuit
        t_ansatz_circ = ansatz.transpiled_circuit(
            self.instance, method=transpiler, enforce_bijection=True)

        # this gives us the final measurements the WPO set requires
        empty_circuit = QuantumCircuit(ansatz.nb_qubits)
        measurements = self._weighted_operators.construct_evaluation_circuit(
            wave_function=empty_circuit,
            statevector_mode=self.instance.is_statevector,
            )

        # convert measurement instructions to the backend's gateset
        simple_instance = QuantumInstance(
            self.instance.backend, optimization_level=0)
        measurements = simple_instance.transpile(measurements)

        # combine the transpiled ansatz circuit with the measurement
        # instructions
        self._meas_circuits = []
        for meas in measurements:
            tmp = t_ansatz_circ.copy()

            # add classical register from measurement circ
            tmp.add_register(meas.cregs[0])

            for instruction in meas.data:

                # this can be used as a simple test of whether the transpile of
                # measurements using `simple_instance` did anything more
                # complicated than changing the gate set
                if isinstance(instruction[0], Measure):
                    if not instruction[1][0].index == instruction[2][0].index:
                        raise ValueError(
                            'Simple transpiler pass in CostWPO constructor has'
                            + ' moved qubits.'
                        )

                # parse quantum and classical args
                qargs = _map_qargs(instruction[1], ansatz.transpiler_map,
                                   tmp.qregs[0])
                cargs = []
                if len(instruction[2]) > 0:
                    cargs = [Clbit(tmp.cregs[0], instruction[2][0].index)]

                # append instruction onto the end of the transpiled ansatz circ
                tmp.append(instruction[0], qargs, cargs)

            # transfer name
            tmp.name = meas.name

            self._meas_circuits.append(tmp)

    def __call__(self, params):
        """
        Wrapper around cost function so it may be called directly

        Parameters
        ----------
        params : array-like
            Params to bind to the ansatz variables (assumed input is same
            length as self.ansatz.nb_params).

        Returns
        -------
        TYPE
            2d array (Same as Cost), Single entery for each each input
            parameter.

        """
        params = np.atleast_2d(params)
        res = []
        for pp in params:
            circs = self.bind_params_to_meas(pp)
            results = self.instance.execute(circs)
            res.append(self.evaluate_cost(results))
        return np.atleast_2d(res).T

    def shot_noise(self, params, nb_shots=8):
        params = np.squeeze(params)
        params = np.atleast_2d([params for ii in range(nb_shots)])
        return self.__call__(params)

    def evaluate_cost_and_std(
        self,
        results: Result,
        name='',
        real_part=True,
        **kwargs,
    ):
        """
        Evaluate the expectation value of the state produced by the ansatz
        against the weighted Pauli operators stored, using the results from an
        experiment.

        NOTE: this takes the statevector mode from the cost obj's quantum
        instance attribute, which is not necessarily the instance that has
        produced the results. The executing instance should be the same backend
        however, and so (hopefully) operate in the same statevector mode.

        Parameters
        ----------
        results : qiskit results obj
            Results to evaluate the operators against
        name : string, optional
            Used to resolve circuit naming
        """
        mean, std = self._weighted_operators.evaluate_with_result(
            results,
            statevector_mode=self.instance.is_statevector,
            circuit_name_prefix=name
            )

        if real_part:
            if (
                (not np.isclose(np.imag(mean), 0.))
                or (not np.isclose(np.imag(std), 0.))
            ):
                print('Warning, `evaluate_cost_and_std` throwing away non-zero'
                      + ' imaginary part.', file=sys.stderr)
            mean = np.real(mean)
            std = np.real(std)

        self.last_evaluation = {
            'mean': mean,
            'std': std,
        }
        return mean, std

    def evaluate_cost(
        self,
        results: Result,
        name='',
        real_part=True,
        **kwargs,
    ):
        """
        Evaluate the expectation value of the state produced by the ansatz
        against the weighted Pauli operators stored, using the results from an
        experiment.

        Parameters
        ----------
        results : qiskit results obj
            Results to evaluate the operators against
        name : string, optional
            Used to resolve circuit naming
        """
        mean, _ = self.evaluate_cost_and_std(
            results=results,
            name=name,
            real_part=real_part,
            **kwargs,
            )
        return mean

    @property
    def _min_energy(self):
        print('warning CostWPO._min_energy is not working')
        eig = ExactEigensolver(self._weighted_operators)
        eig = eig.run()
        return np.squeeze(abs(eig.eigenvalues))


class CostWPOquimb(CostWPO):
    """
    """

    def bind_params_to_meas(self, params=None, params_names=None):
        """
        """
        return [{params_names: params}]

    def evaluate_cost(
        self,
        results,
        name='',
        real_part=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        results : dict
            Pairs name: pt
        """

        # bind ansatz circuit at current param point
        param_pt = results[name]
        bound_circ = self.ansatz.circuit.bind_parameters(
            dict(zip(self.ansatz.params, param_pt)))
        # convert to TN
        tn_of_ansatz = qTNfromQASM(bound_circ.qasm())
        # first time need to unpack wpo into quimb form
        if not hasattr(self, 'measurement_ops'):
            # total hack and reliant on exact form of `.print_details()`
            # string, but works currently
            self._pauli_weights = [
                np.complex128(lne.split('\t')[1])
                for lne in self._weighted_operators.print_details().split('\n')
                if len(lne) > 0 and not lne[0] == 'T'
            ]
            pauli_strings = [
                lne.split('\t')[0]
                for lne in self._weighted_operators.print_details().split('\n')
                if len(lne) > 0 and not lne[0] == 'T'
            ]
            self._measurement_ops = [
                qu.kron(*[qu.pauli(i) for i in p]) for p in pauli_strings
            ]

        mean = np.sum(
            np.real(
                np.array(tn_of_ansatz.local_expectation(
                    self._measurement_ops,
                    where=tuple(range(self.ansatz.nb_qubits)),)
                ) * self._pauli_weights
            )
        )
        self.last_evaluation = {
            'mean': mean,
            'std': 0.,
        }
        return mean

    def evaluate_cost_and_std(
        self,
        results,
        name='',
        real_part=True,
    ):
        """
        """
        return self.evaluate_cost(results, name, real_part,), 0.
