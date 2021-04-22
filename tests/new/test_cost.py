"""
Tests for cost classes and functions
"""

import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.aqua import QuantumInstance

from qcoptim.ansatz import TrivialAnsatz, RandomAnsatz
from qcoptim.cost import CrossFidelity
from qcoptim.utilities import RandomMeasurementHandler


def test_cross_fidelity():
    """ """
    num_qubits = 2
    num_random = 500
    seed = 0

    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))

    circ = QuantumCircuit(num_qubits)
    circ_ortho = QuantumCircuit(num_qubits)
    circ_ortho.x(0)
    circ_superpos = QuantumCircuit(num_qubits)
    circ_superpos.h(0)

    # get one-side of the data
    init_crossfid = CrossFidelity(
        TrivialAnsatz(circ),
        instance,
        nb_random=num_random,
        seed=seed,
        prefixA='init-data',
    )
    circs = init_crossfid.bind_params_to_meas([])
    assert len(circs) == num_random
    results = instance.execute(circs)
    comparison_results = init_crossfid.tag_results_metadata(results)

    # new objs to compute cross-fidelity overlaps
    crossfid = CrossFidelity(
        TrivialAnsatz(circ),
        instance,
        nb_random=num_random,
        seed=seed,
        comparison_results=comparison_results,
        prefixA='new-data',
        prefixB='init-data',
    )
    crossfid_ortho = CrossFidelity(
        TrivialAnsatz(circ_ortho),
        instance,
        nb_random=num_random,
        seed=seed,
        comparison_results=comparison_results,
        prefixA='ortho-data',
        prefixB='init-data',
    )
    crossfid_superpos = CrossFidelity(
        TrivialAnsatz(circ_superpos),
        instance,
        nb_random=num_random,
        seed=seed,
        comparison_results=comparison_results,
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

    # compute overlaps and test values
    results = instance.execute(circs)
    same, same_std = crossfid.evaluate_cost_and_std(results)
    ortho, ortho_std = crossfid_ortho.evaluate_cost_and_std(results)
    superpos, superpos_std = crossfid_superpos.evaluate_cost_and_std(results)
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

    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    ansatz = RandomAnsatz(2, 2)
    rand_meas_handler = RandomMeasurementHandler(
        ansatz, instance, num_random, seed=seed, circ_name=circ_name,
    )

    # get one-side of the data
    init_crossfid = CrossFidelity(
        ansatz,
        instance,
        rand_meas_handler=rand_meas_handler,
    )
    point = np.random.random(ansatz.nb_params)
    circs = init_crossfid.bind_params_to_meas(point)
    assert len(circs) == num_random
    results = instance.execute(circs)
    comparison_results = init_crossfid.tag_results_metadata(results)

    # make subsampled obj
    crossfid = CrossFidelity(
        ansatz,
        instance,
        nb_random=num_random//2,
        rand_meas_handler=rand_meas_handler,
        comparison_results=comparison_results,
        prefixB=circ_name(''),
    )

    # check full number of circuits yielded
    point = np.random.random(ansatz.nb_params)
    circs = crossfid.bind_params_to_meas(point)
    assert len(circs) == num_random

    # check that only need first half to evaluate crossfid
    results = instance.execute(circs[:num_random//2])
    _, _ = crossfid.evaluate_cost_and_std(results)


def test_cross_fidelity_shared_rand_meas_handler():
    """ """
    num_random = 10
    seed = 0

    def circ_name(idx):
        return 'test_circ'+f'{idx}'

    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    ansatz = RandomAnsatz(2, 2)
    rand_meas_handler = RandomMeasurementHandler(
        ansatz, instance, num_random, seed=seed, circ_name=circ_name,
    )

    crossfid1 = CrossFidelity(
        ansatz,
        instance,
        rand_meas_handler=rand_meas_handler,
    )
    crossfid2 = CrossFidelity(
        ansatz,
        instance,
        rand_meas_handler=rand_meas_handler,
    )

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
