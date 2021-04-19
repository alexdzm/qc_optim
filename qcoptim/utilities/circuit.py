"""
"""

import numpy as np

from qiskit.quantum_info import random_unitary


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
