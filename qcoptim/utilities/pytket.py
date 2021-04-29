"""
"""

from qiskit.aqua.utils.backend_utils import is_ibmq_provider

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit, IBMQBackend


def compile_for_backend(backend, circuit):
    """
    Use pytket to compile single circuit or list of circuits for a IBMQ
    backend, preserves circuit names.

    Parameters
    ----------
    backend : qiskit backend
        IBMQ backend to compile circuits for
    circuit : qiskit.QuantumCircuit, list(qiskit.QuantumCircuit)
        Circuit or list circuits to compile

    Returns
    -------
    qiskit.QuantumCircuit, list(qiskit.QuantumCircuit)
        return matches format of arg
    """
    if not is_ibmq_provider(backend):
        return circuit

    pytket_backend = IBMQBackend(backend.name(),
                                 hub=backend.hub,
                                 group=backend.group,
                                 project=backend.project,)

    single_circ = False
    if not isinstance(circuit, list):
        single_circ = True
        circuit = [circuit]

    transpiled_circuits = []
    for circ in circuit:
        pytket_circuit = qiskit_to_tk(circ)
        pytket_backend.compile_circuit(pytket_circuit, optimisation_level=2)
        transpiled_circuits.append(tk_to_qiskit(pytket_circuit))

        # preserve circuit name
        transpiled_circuits[-1].name = circ.name

    if single_circ:
        return transpiled_circuits[0]
    return transpiled_circuits
