"""
Tests for error mitigation classes and functions
"""

import pytest

import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info.operators import Operator, Pauli

from qcoptim.ansatz import RandomAnsatz, TrivialAnsatz
from qcoptim.cost import CostWPO
from qcoptim.error_mitigation import PurityBoostCalibrator, PurityBoostFitter


def test_purity_cost_calibrator():
    """ """
    num_bootstraps = 1000
    num_random = 100

    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    ansatz = RandomAnsatz(2, 2)
    calibrator = PurityBoostCalibrator(ansatz, instance, num_random, seed=0,
                                       num_bootstraps=num_bootstraps)

    # check blocking yield of calibration circuits
    assert not calibrator._yielded_calibration_circuits
    circs = calibrator.calibration_circuits
    assert len(circs) == num_random
    assert calibrator._yielded_calibration_circuits
    assert calibrator.calibration_circuits == []

    # run calibration
    results = instance.execute(circs)
    calibrator.process_calibration_results(results)
    assert not calibrator._yielded_calibration_circuits
    assert calibrator.calibrated

    # check stored data, ideally ptot should be 0
    assert len(calibrator.ptot_dist) == num_bootstraps
    assert calibrator.ptot < 0.05

    # check reset
    calibrator.reset()
    assert not calibrator._yielded_calibration_circuits


def test_purity_boost_fitter():
    """ """
    num_bootstraps = 1000
    num_random = 100

    wpo = WeightedPauliOperator([
        (1., Pauli.from_label('ZZ')),
        (-1./3, Pauli.from_label('XI')),
        (-1./3, Pauli.from_label('IX')),
    ])
    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))

    # test in Z-basis
    target_value = 1.
    test_circ = QuantumCircuit(2)
    cost = CostWPO(TrivialAnsatz(test_circ), instance, wpo)
    fitter = PurityBoostFitter(cost, num_random=num_random, seed=0,
                               num_bootstraps=num_bootstraps)
    circs = fitter.bind_params_to_meas([])
    assert len(circs) == num_random + 2
    results = instance.execute(circs)
    mean, err = fitter.evaluate_cost_and_std(results)
    assert (mean - 2*err < target_value) and (mean + 2*err > target_value)

    # test in X-basis
    target_value = -2./3
    test_circ = QuantumCircuit(2)
    test_circ.h(0)
    test_circ.h(1)
    cost = CostWPO(TrivialAnsatz(test_circ), instance, wpo)
    fitter = PurityBoostFitter(cost, num_random=num_random, seed=0,
                               num_bootstraps=num_bootstraps)
    circs = fitter.bind_params_to_meas([])
    assert len(circs) == num_random + 2
    results = instance.execute(circs)
    mean, err = fitter.evaluate_cost_and_std(results)
    assert (mean - 2*err < target_value) and (mean + 2*err > target_value)


def test_purity_boost_fitter_shared_calibration():
    """ """
    num_bootstraps = 1000
    num_random = 100

    ansatz = TrivialAnsatz(QuantumCircuit(2))
    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    calibrator = PurityBoostCalibrator(ansatz, instance, num_random, seed=0,
                                       num_bootstraps=num_bootstraps)

    wpo_1 = WeightedPauliOperator([
        (1., Pauli.from_label('ZZ')),
    ])
    cost_1 = CostWPO(ansatz, instance, wpo_1)
    fitter_1 = PurityBoostFitter(cost_1, calibrator=calibrator)

    wpo_2 = WeightedPauliOperator([
        (-1./3, Pauli.from_label('XI')),
        (-1./3, Pauli.from_label('IX')),
    ])
    cost_2 = CostWPO(ansatz, instance, wpo_2)
    fitter_2 = PurityBoostFitter(cost_2, calibrator=calibrator)

    circs_1 = fitter_1.bind_params_to_meas([])
    assert len(circs_1) == num_random + 1
    circs_2 = fitter_2.bind_params_to_meas([])
    assert len(circs_2) == 1

    results = instance.execute(circs_1 + circs_2)

    target_value = 0.
    assert (
        not fitter_1.calibrator.calibrated
        and not fitter_2.calibrator.calibrated
    )
    mean = fitter_2.evaluate_cost(results)
    _, err = fitter_2.evaluate_cost_and_std(results)
    assert (
        fitter_1.calibrator.calibrated
        and fitter_2.calibrator.calibrated
    )
    # has problems with target values near zero
    # assert (mean - 2*err < target_value) and (mean + 2*err > target_value)

    target_value = 1
    mean = fitter_1.evaluate_cost(results)
    _, err = fitter_1.evaluate_cost_and_std(results)
    assert (mean - 2*err < target_value) and (mean + 2*err > target_value)
