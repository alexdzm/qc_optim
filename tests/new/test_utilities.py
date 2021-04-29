"""
Tests for utilities
"""

import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.aqua import QuantumInstance

from qcoptim.ansatz import RandomAnsatz, TrivialAnsatz
from qcoptim.utilities import (
    RandomMeasurementHandler,
)
from qcoptim.utilities.pytket import compile_for_backend


def test_pytket_compile_for_backend():
    """  """


def test_random_measurement_handler():
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

    point = np.ones(ansatz.nb_params)
    circs = rand_meas_handler.circuits(point)
    assert len(circs) == num_random
    assert circs[0].name == 'test_circ0'

    # test that no circuits are returned on second request
    assert len(rand_meas_handler.circuits(point)) == 0

    # test reset
    rand_meas_handler.reset()
    assert len(rand_meas_handler.circuits(point)) == num_random
    assert rand_meas_handler.circuits(point) == []

    # change circuit name func
    rand_meas_handler.circ_name = lambda idx: 'HaarRandom' + f'{idx}'

    # test that new point releases lock
    point2 = 2. * np.ones(ansatz.nb_params)
    circs2 = rand_meas_handler.circuits(point2)
    assert len(circs2) == num_random
    assert circs2[0].name == 'HaarRandom0'


def test_random_measurement_handler_trivial_ansatz():
    """
    Test special case of ansatz with no parameters
    """
    num_random = 10
    seed = 0

    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    circ = QuantumCircuit(2)
    rand_meas_handler = RandomMeasurementHandler(
        TrivialAnsatz(circ), instance, num_random, seed=seed,
    )

    circs = rand_meas_handler.circuits([])
    assert len(circs) == num_random
    assert rand_meas_handler.circuits([]) == []


def test_random_measurement_handler_2d_point():
    """
    Test correct behaviour with array of points
    """
    num_random = 10
    num_points = 3
    seed = 0

    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    ansatz = RandomAnsatz(2, 2)
    rand_meas_handler = RandomMeasurementHandler(
        ansatz, instance, num_random, seed=seed,
    )

    points = np.random.random(num_points*ansatz.nb_params)
    points = points.reshape((num_points, ansatz.nb_params))
    circs = rand_meas_handler.circuits(points)
    assert len(circs) == num_random*num_points
    assert rand_meas_handler.circuits(points) == []

    # change one of points, should unlock
    points[0, :] = 0.
    circs = rand_meas_handler.circuits(points)
    assert len(circs) == num_random*num_points
    assert rand_meas_handler.circuits(points) == []
