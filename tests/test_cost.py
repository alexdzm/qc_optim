"""
Tests for cost classes and functions
"""

import pytest

import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.aqua import QuantumInstance

from qcoptim.ansatz import TrivialAnsatz, RandomAnsatz
from qcoptim.cost import CrossFidelity
from qcoptim.utilities import (
    RandomMeasurementHandler,
    make_quantum_instance,
    ProcessedResult,
)

_TEST_IBMQ_BACKEND = 'ibmq_santiago'
_TRANSPILER = 'pytket'


@pytest.fixture
def crossfid_test_assets():
    """ """
    num_qubits = 2
    num_random = 100
    seed = 0

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)
    exe_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))

    circ = QuantumCircuit(num_qubits)
    circ_ortho = QuantumCircuit(num_qubits)
    circ_ortho.x(0)
    circ_superpos = QuantumCircuit(num_qubits)
    circ_superpos.h(0)

    # get one-side of the data
    init_crossfid = CrossFidelity(
        TrivialAnsatz(circ),
        transpile_instance,
        transpiler=_TRANSPILER,
        nb_random=num_random,
        seed=seed,
        prefixA='init-data',
    )
    circs = init_crossfid.bind_params_to_meas([])
    results1 = exe_instance.execute(circs)
    # comparison_results = init_crossfid.tag_results_metadata(results)

    # new objs to compute cross-fidelity overlaps
    crossfid = CrossFidelity(
        TrivialAnsatz(circ),
        transpile_instance,
        transpiler=_TRANSPILER,
        nb_random=num_random,
        seed=seed,
        prefixA='new-data',
        prefixB='init-data',
    )
    crossfid_ortho = CrossFidelity(
        TrivialAnsatz(circ_ortho),
        transpile_instance,
        transpiler=_TRANSPILER,
        nb_random=num_random,
        seed=seed,
        prefixA='ortho-data',
        prefixB='init-data',
    )
    crossfid_superpos = CrossFidelity(
        TrivialAnsatz(circ_superpos),
        transpile_instance,
        transpiler=_TRANSPILER,
        nb_random=num_random,
        seed=seed,
        prefixA='superpos-data',
        prefixB='init-data',
    )
    circs = crossfid.bind_params_to_meas([])
    circs += crossfid_ortho.bind_params_to_meas([])
    circs += crossfid_superpos.bind_params_to_meas([])

    # compute overlaps and test values
    results2 = exe_instance.execute(circs)

    return crossfid, crossfid_ortho, crossfid_superpos, results1, results2


def test_cross_fidelity_circuit_counts():
    """ """
    num_qubits = 2
    num_random = 10
    seed = 0

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)
    # exe_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))

    circ = QuantumCircuit(num_qubits)
    circ_ortho = QuantumCircuit(num_qubits)
    circ_ortho.x(0)
    circ_superpos = QuantumCircuit(num_qubits)
    circ_superpos.h(0)

    # get one-side of the data
    init_crossfid = CrossFidelity(
        TrivialAnsatz(circ),
        transpile_instance,
        transpiler=_TRANSPILER,
        nb_random=num_random,
        seed=seed,
        prefixA='init-data',
    )
    circs = init_crossfid.bind_params_to_meas([])
    assert len(circs) == num_random

    # new objs to compute cross-fidelity overlaps
    crossfid = CrossFidelity(
        TrivialAnsatz(circ),
        transpile_instance,
        transpiler=_TRANSPILER,
        nb_random=num_random,
        seed=seed,
        prefixA='new-data',
        prefixB='init-data',
    )
    crossfid_ortho = CrossFidelity(
        TrivialAnsatz(circ_ortho),
        transpile_instance,
        transpiler=_TRANSPILER,
        nb_random=num_random,
        seed=seed,
        prefixA='ortho-data',
        prefixB='init-data',
    )
    crossfid_superpos = CrossFidelity(
        TrivialAnsatz(circ_superpos),
        transpile_instance,
        transpiler=_TRANSPILER,
        nb_random=num_random,
        seed=seed,
        prefixA='superpos-data',
        prefixB='init-data',
    )
    circs = crossfid.bind_params_to_meas([])
    assert len(circs) == num_random
    circs += crossfid_ortho.bind_params_to_meas([])
    assert len(circs) == 2*num_random
    circs += crossfid_superpos.bind_params_to_meas([])
    assert len(circs) == 3*num_random

    # test double request lock
    assert crossfid.bind_params_to_meas([]) == []
    assert crossfid_ortho.bind_params_to_meas([]) == []
    assert crossfid_superpos.bind_params_to_meas([]) == []


@pytest.mark.parametrize("process_result", [False, True])
def test_cross_fidelity_nobootstrapping(crossfid_test_assets, process_result):
    """ """

    (crossfid, crossfid_ortho, crossfid_superpos, 
     results1, results2) = crossfid_test_assets

    if process_result:
        results1 = ProcessedResult(results1)
    if process_result:
        results2 = ProcessedResult(results2)

    # set comparison results
    crossfid.comparison_results = results1
    crossfid_ortho.comparison_results = results1
    crossfid_superpos.comparison_results = results1

    # set bootstrapping
    num_bootstraps = 0
    crossfid._num_bootstraps = num_bootstraps
    crossfid_ortho._num_bootstraps = num_bootstraps
    crossfid_superpos._num_bootstraps = num_bootstraps

    same, same_std = crossfid.evaluate_cost_and_std(
        results2, vectorise=True)
    tmp1, tmp2 = crossfid.evaluate_cost_and_std(
        results2, vectorise=False)
    assert np.isclose(same, tmp1)  # , atol=0.01
    assert np.isclose(same_std, tmp2)  # , atol=0.01

    ortho, ortho_std = crossfid_ortho.evaluate_cost_and_std(
        results2, vectorise=True)
    tmp1, tmp2 = crossfid_ortho.evaluate_cost_and_std(
        results2, vectorise=False)
    assert np.isclose(ortho, tmp1)  # , atol=0.01
    assert np.isclose(ortho_std, tmp2)  # , atol=0.01

    superpos, superpos_std = crossfid_superpos.evaluate_cost_and_std(
        results2, vectorise=True)
    tmp1, tmp2 = crossfid_superpos.evaluate_cost_and_std(
        results2, vectorise=False)
    assert np.isclose(superpos, tmp1)  # , atol=0.01
    assert np.isclose(superpos_std, tmp2)  # , atol=0.01

    for mean, std, target in zip(
        [same, ortho, superpos],
        [same_std, ortho_std, superpos_std],
        [1., 0., 0.5]
    ):
        np.isclose(mean, target, atol=0.1)
        # assert mean - 4*std < target
        # assert mean + 4*std > target


@pytest.mark.parametrize("process_result", [False, True])
def test_cross_fidelity_bootstrapping(crossfid_test_assets, process_result):
    """ """

    (crossfid, crossfid_ortho, crossfid_superpos,
     results1, results2) = crossfid_test_assets

    if process_result:
        results1 = ProcessedResult(results1)
    if process_result:
        results2 = ProcessedResult(results2)

    # set comparison results
    crossfid.comparison_results = results1
    crossfid_ortho.comparison_results = results1
    crossfid_superpos.comparison_results = results1

    # set bootstrapping
    num_bootstraps = 1000
    crossfid._num_bootstraps = num_bootstraps
    crossfid_ortho._num_bootstraps = num_bootstraps
    crossfid_superpos._num_bootstraps = num_bootstraps

    same, same_std = crossfid.evaluate_cost_and_std(
        results2, vectorise=True)
    tmp1, tmp2 = crossfid.evaluate_cost_and_std(
        results2, vectorise=False)
    assert np.isclose(same, tmp1, atol=1E-4)  # , atol=0.01
    assert np.isclose(same_std, tmp2, atol=1E-4)  # , atol=0.01

    ortho, ortho_std = crossfid_ortho.evaluate_cost_and_std(
        results2, vectorise=True)
    tmp1, tmp2 = crossfid_ortho.evaluate_cost_and_std(
        results2, vectorise=False)
    assert np.isclose(ortho, tmp1, atol=1E-4)  # , atol=0.01
    assert np.isclose(ortho_std, tmp2, atol=1E-4)  # , atol=0.01

    superpos, superpos_std = crossfid_superpos.evaluate_cost_and_std(
        results2, vectorise=True)
    tmp1, tmp2 = crossfid_superpos.evaluate_cost_and_std(
        results2, vectorise=False)
    assert np.isclose(superpos, tmp1, atol=1E-4)  # , atol=0.01
    assert np.isclose(superpos_std, tmp2, atol=1E-4)  # , atol=0.01

    for mean, std, target in zip(
        [same, ortho, superpos],
        [same_std, ortho_std, superpos_std],
        [1., 0., 0.5]
    ):
        np.isclose(mean, target, atol=0.1)
        # assert mean - 4*std < target
        # assert mean + 4*std > target


def test_cross_fidelity_subsampling_rand_meas_handler():
    """ """
    num_random = 10
    seed = 0

    def circ_name(idx):
        return 'test_circ'+f'{idx}'

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)
    exe_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))

    ansatz = RandomAnsatz(2, 2, strict_transpile=True)
    rand_meas_handler = RandomMeasurementHandler(
        ansatz,
        transpile_instance,
        num_random,
        transpiler=_TRANSPILER,
        seed=seed,
        circ_name=circ_name,
    )

    # get one-side of the data
    init_crossfid = CrossFidelity(ansatz, rand_meas_handler=rand_meas_handler,)
    point = np.random.random(ansatz.nb_params)
    circs = init_crossfid.bind_params_to_meas(point)
    assert len(circs) == num_random
    results = exe_instance.execute(circs)
    # comparison_results = init_crossfid.tag_results_metadata(results)

    # make subsampled obj
    crossfid = CrossFidelity(
        ansatz,
        nb_random=num_random//2,
        rand_meas_handler=rand_meas_handler,
        comparison_results=results,
        prefixB=circ_name(''),
    )

    # check full number of circuits yielded
    point = np.random.random(ansatz.nb_params)
    circs = crossfid.bind_params_to_meas(point)
    assert len(circs) == num_random

    # check that only need first half to evaluate crossfid
    results = exe_instance.execute(circs[:num_random//2])
    _, _ = crossfid.evaluate_cost_and_std(results)


def test_cross_fidelity_shared_rand_meas_handler():
    """ """
    num_random = 10
    seed = 0

    def circ_name(idx):
        return 'test_circ'+f'{idx}'

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)
    ansatz = RandomAnsatz(2, 2, strict_transpile=True)
    rand_meas_handler = RandomMeasurementHandler(
        ansatz,
        transpile_instance,
        num_random,
        transpiler=_TRANSPILER,
        seed=seed,
        circ_name=circ_name,
    )

    crossfid1 = CrossFidelity(ansatz, rand_meas_handler=rand_meas_handler,)
    crossfid2 = CrossFidelity(ansatz, rand_meas_handler=rand_meas_handler,)

    # check blocking behaviour of cross-fid objs with shared rand_meas_handler
    point = np.ones(ansatz.nb_params)
    circs = crossfid1.bind_params_to_meas(point)
    assert len(circs) == num_random
    assert crossfid2.bind_params_to_meas(point) == []
    assert circs[0].name == circ_name(0)

    # check requesting a different point releases lock
    point2 = 2*np.ones(ansatz.nb_params)
    circs = crossfid2.bind_params_to_meas(point2)
    assert len(circs) == num_random
    assert crossfid1.bind_params_to_meas(point2) == []
