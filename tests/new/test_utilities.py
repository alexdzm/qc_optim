"""
Tests for utilities
"""

import numpy as np

from qiskit import Aer
from qiskit.aqua import QuantumInstance

from qcoptim.ansatz import RandomAnsatz
from qcoptim.utilities import (
    RandomMeasurementHandler,
)


def test_random_measurement_handler():
    """ """
    num_random = 500
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
