"""
"""

from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance

from qcoptim.ansatz import TrivialAnsatz
from qcoptim.utilities import make_quantum_instance


def test_ansatz_transpile():
    """
    Transpile code is in the BaseAnsatz class, but that is not directly
    instanceable. TrivialAnsatz is the simplest derived class so that is used
    """
    # simple circuit linking all qubits
    circ = QuantumCircuit(4)
    circ.h(0)
    circ.cx(0, 1)
    circ.h(2)
    circ.cx(2, 3)
    circ.cx(1, 2)
    ansatz = TrivialAnsatz(circ)

    sim_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    ibmq_instance = make_quantum_instance('ibmq_santiago')

    # should raise AttributeError
    try:
        _ = ansatz.transpiler_map
        assert False, 'should have raised error'
    except AttributeError:
        pass

    # test these all work
    _ = ansatz.transpiled_circuit(ibmq_instance, engine='instance',
                                  enforce_bijection=True)
    _ = ansatz.transpiler_map
    _ = ansatz.transpiled_circuit(ibmq_instance, engine='pytket',
                                  enforce_bijection=True)
    _ = ansatz.transpiler_map

    # these should fail
    try:
        ansatz.transpiled_circuit(ibmq_instance, engine='instance',
                                  strict=True)
        assert False, 'should have raised error'
    except ValueError:
        pass
    try:
        ansatz.transpiled_circuit(sim_instance, engine='pytket',
                                  strict=True)
        assert False, 'should have raised error'
    except ValueError:
        pass
