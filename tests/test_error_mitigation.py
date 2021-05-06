"""
Tests for error mitigation classes and functions, run with pytest.
"""

import pytest

import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info.operators import Pauli

from qcoptim.ansatz import RandomAnsatz, TrivialAnsatz
from qcoptim.utilities import RandomMeasurementHandler, make_quantum_instance
from qcoptim.cost import CostWPO, CrossFidelity
from qcoptim.error_mitigation import (
    multiply_cx,
    CXMultiplierFitter,
    PurityBoostCalibrator,
    PurityBoostFitter,
    MultiStrategyFitter,
)

_TEST_IBMQ_BACKEND = 'ibmq_santiago'
_TRANSPILERS = ['instance', 'pytket']


def test_multiply_cx():
    """ """
    qc1 = QuantumCircuit(3)
    qc3 = qc1.copy()
    qc5 = qc1.copy()

    qc1.h(0)
    qc1.cx(0, 1)
    qc1.cx(1, 2)

    qc3.h(0)
    for _ in range(3):
        qc3.cx(0, 1)
    for _ in range(3):
        qc3.cx(1, 2)

    qc5.h(0)
    for _ in range(5):
        qc5.cx(0, 1)
    for _ in range(5):
        qc5.cx(1, 2)

    assert multiply_cx(qc1, 1) == qc1
    assert multiply_cx(qc1, 3) == qc3
    assert multiply_cx(qc1, 5) == qc5


@pytest.mark.parametrize("transpiler", _TRANSPILERS)
@pytest.mark.parametrize("extrapolation_strategy", ['richardson', 'linear'])
def test_cx_multiplier_fitter(transpiler, extrapolation_strategy):
    """ """
    max_factor = 7
    allowed_standard_errors = 4
    if extrapolation_strategy == 'linear':
        # linear typically performs poorly
        allowed_standard_errors = 10

    wpo = WeightedPauliOperator([
        (1., Pauli.from_label('ZZ')),
        (-1./3, Pauli.from_label('XI')),
        (-1./3, Pauli.from_label('IX')),
    ])

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)
    exe_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                   shots=8192)

    # test in Z-basis
    target_value = 1.
    test_circ = QuantumCircuit(2)
    test_circ.cx(0, 1)
    cost = CostWPO(
        TrivialAnsatz(test_circ),
        transpile_instance,
        wpo,
        transpiler=transpiler,
    )
    fitter = CXMultiplierFitter(
        cost, max_factor,
        extrapolation_strategy=extrapolation_strategy,
    )
    circs = fitter.bind_params_to_meas([])
    assert len(circs) == 2 * ((max_factor + 1) // 2)
    results = exe_instance.execute(circs)
    mean = fitter.evaluate_cost(results)
    _, err = fitter.evaluate_cost_and_std(results)
    assert mean - allowed_standard_errors*err < target_value
    assert mean + allowed_standard_errors*err > target_value

    # test in X-basis
    target_value = -2./3
    test_circ = QuantumCircuit(2)
    test_circ.cx(0, 1)
    test_circ.h(0)
    test_circ.h(1)
    cost = CostWPO(
        TrivialAnsatz(test_circ),
        transpile_instance,
        wpo,
        transpiler=transpiler,
    )
    fitter = CXMultiplierFitter(
        cost, max_factor,
        extrapolation_strategy=extrapolation_strategy,
    )
    circs = fitter.bind_params_to_meas([])
    assert len(circs) == 2 * ((max_factor + 1) // 2)
    results = exe_instance.execute(circs)
    mean = fitter.evaluate_cost(results)
    _, err = fitter.evaluate_cost_and_std(results)
    assert mean - allowed_standard_errors*err < target_value
    assert mean + allowed_standard_errors*err > target_value


@pytest.mark.parametrize("transpiler", _TRANSPILERS)
def test_purity_boost_calibrator(transpiler):
    """ """
    num_bootstraps = 1000
    num_random = 100

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)
    exe_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))

    ansatz = RandomAnsatz(2, 2, strict_transpile=True)
    calibrator = PurityBoostCalibrator(
        ansatz,
        transpile_instance,
        num_random,
        transpiler=transpiler,
        seed=0,
        num_bootstraps=num_bootstraps)

    # check blocking yield of calibration circuits
    circs = calibrator.calibration_circuits
    assert len(circs) == num_random
    assert calibrator.calibration_circuits == []

    # run calibration
    results = exe_instance.execute(circs)
    calibrator.process_calibration_results(results)
    assert calibrator.calibrated

    # check stored data, ideally ptot should be 0
    target_value = 0
    assert calibrator.ptot - 4*calibrator.ptot_std < target_value
    assert calibrator.ptot + 4*calibrator.ptot_std > target_value

    # check reset
    calibrator.reset()
    assert len(calibrator.calibration_circuits) == num_random
    assert calibrator.calibration_circuits == []

    # changing calibration point will also cause reset
    calibrator.calibration_point = np.ones(ansatz.nb_params)
    assert len(calibrator.calibration_circuits) == num_random
    assert calibrator.calibration_circuits == []


@pytest.mark.parametrize("transpiler", _TRANSPILERS)
def test_purity_boost_calibrator_subsampled_rand_meas_handler(transpiler):
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
        transpiler=transpiler,
        seed=seed,
        circ_name=circ_name,
    )

    calibration_point = np.ones(ansatz.nb_params)
    calibrator = PurityBoostCalibrator(
        ansatz, num_random=num_random//2,
        rand_meas_handler=rand_meas_handler,
        calibration_point=calibration_point,
    )

    # check full number of circuits yielded
    circs = calibrator.calibration_circuits
    assert len(circs) == num_random

    # check that only need first half to evaluate crossfid
    results = exe_instance.execute(circs[:num_random//2])
    calibrator.process_calibration_results(results)


@pytest.mark.parametrize("transpiler", _TRANSPILERS)
def test_purity_boost_calibrator_shared_rand_meas_handler(transpiler):
    """ """
    num_random = 500
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
        transpiler=transpiler,
        seed=seed,
        circ_name=circ_name,
    )

    calibration_point = np.ones(ansatz.nb_params)
    calibrator1 = PurityBoostCalibrator(
        ansatz, rand_meas_handler=rand_meas_handler,
        calibration_point=calibration_point,
    )
    calibrator2 = PurityBoostCalibrator(
        ansatz, rand_meas_handler=rand_meas_handler,
        calibration_point=calibration_point,
    )

    # check blocking behaviour of calibrators with shared rand_meas_handler
    circs = calibrator1.calibration_circuits
    assert len(circs) == num_random
    assert calibrator2.calibration_circuits == []

    # run both calibrations
    results = exe_instance.execute(circs)
    calibrator1.process_calibration_results(results)
    assert calibrator1.calibrated
    calibrator2.process_calibration_results(results)
    assert calibrator2.calibrated

    # check stored data, ideally ptot should be 0
    target_value = 0
    for calib in [calibrator1, calibrator2]:
        assert calib.ptot - 4*calib.ptot_std < target_value
        assert calib.ptot + 4*calib.ptot_std > target_value

    # should have slightly different values, from bootstrapping
    assert not np.isclose(calibrator1.ptot_std, calibrator2.ptot_std)


@pytest.mark.parametrize("transpiler", _TRANSPILERS)
def test_purity_boost_fitter(transpiler):
    """ """
    num_bootstraps = 1000
    num_random = 100

    wpo = WeightedPauliOperator([
        (1., Pauli.from_label('ZZ')),
        (-1./3, Pauli.from_label('XI')),
        (-1./3, Pauli.from_label('IX')),
    ])

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)
    exe_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))

    # test in Z-basis
    target_value = 1.
    test_circ = QuantumCircuit(2)
    cost = CostWPO(
        TrivialAnsatz(test_circ),
        transpile_instance,
        wpo,
        transpiler=transpiler,
    )
    fitter = PurityBoostFitter(cost, num_random=num_random, seed=0,
                               num_bootstraps=num_bootstraps)
    circs = fitter.bind_params_to_meas([])
    assert len(circs) == num_random + 2
    results = exe_instance.execute(circs)
    mean = fitter.evaluate_cost(results)
    _, err = fitter.evaluate_cost_and_std(results)
    assert mean - 4*err < target_value
    assert mean + 4*err > target_value

    # test in X-basis
    target_value = -2./3
    test_circ = QuantumCircuit(2)
    test_circ.h(0)
    test_circ.h(1)
    cost = CostWPO(
        TrivialAnsatz(test_circ),
        transpile_instance,
        wpo,
        transpiler=transpiler,
    )
    fitter = PurityBoostFitter(cost, num_random=num_random, seed=0,
                               num_bootstraps=num_bootstraps)
    circs = fitter.bind_params_to_meas([])
    assert len(circs) == num_random + 2
    results = exe_instance.execute(circs)
    mean = fitter.evaluate_cost(results)
    _, err = fitter.evaluate_cost_and_std(results)
    assert mean - 4*err < target_value
    assert mean + 4*err > target_value


@pytest.mark.parametrize("transpiler", _TRANSPILERS)
def test_purity_boost_fitter_shared_calibration(transpiler):
    """ """
    num_bootstraps = 1000
    num_random = 100

    ansatz = TrivialAnsatz(QuantumCircuit(2), strict_transpile=True)

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)
    exe_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))

    calibrator = PurityBoostCalibrator(
        ansatz,
        transpile_instance,
        num_random,
        transpiler=transpiler,
        seed=0,
        num_bootstraps=num_bootstraps,
    )

    wpo_1 = WeightedPauliOperator([
        (1., Pauli.from_label('ZZ')),
    ])
    cost_1 = CostWPO(
        ansatz,
        transpile_instance,
        wpo_1,
        transpiler=transpiler,
    )
    fitter_1 = PurityBoostFitter(cost_1, calibrator=calibrator)

    wpo_2 = WeightedPauliOperator([
        (-1./3, Pauli.from_label('XI')),
        (-1./3, Pauli.from_label('IX')),
    ])
    cost_2 = CostWPO(
        ansatz,
        transpile_instance,
        wpo_2,
        transpiler=transpiler,
    )
    fitter_2 = PurityBoostFitter(cost_2, calibrator=calibrator)

    circs_1 = fitter_1.bind_params_to_meas([])
    assert len(circs_1) == num_random + 1
    circs_2 = fitter_2.bind_params_to_meas([])
    assert len(circs_2) == 1

    results = exe_instance.execute(circs_1 + circs_2)

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
    # assert (mean - 4*err < target_value) and (mean + 4*err > target_value)

    target_value = 1
    mean = fitter_1.evaluate_cost(results)
    _, err = fitter_1.evaluate_cost_and_std(results)
    assert mean - 4*err < target_value
    assert mean + 4*err > target_value


@pytest.mark.parametrize("transpiler", _TRANSPILERS)
def test_purity_boost_fitter_sharing_with_crossfid(transpiler):
    """ """
    num_random = 10
    seed = 0

    def circ_name(idx):
        return 'test_circ'+f'{idx}'

    ansatz = RandomAnsatz(2, 2, strict_transpile=True)

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)
    exe_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))

    rand_meas_handler = RandomMeasurementHandler(
        ansatz,
        transpile_instance,
        num_random,
        transpiler=transpiler,
        seed=seed,
        circ_name=circ_name,
    )

    # evaluation point
    point = np.ones(ansatz.nb_params)

    # create cost fitter obj
    calibrator = PurityBoostCalibrator(
        ansatz, rand_meas_handler=rand_meas_handler,
        calibration_point=point,
    )
    wpo = WeightedPauliOperator([
        (1., Pauli.from_label('ZZ')),
    ])
    cost = CostWPO(ansatz, transpile_instance, wpo, transpiler=transpiler)
    fitter = PurityBoostFitter(cost, calibrator=calibrator)

    # cross fid obj using same rand_meas_handler
    crossfid = CrossFidelity(
        ansatz,
        transpile_instance,
        rand_meas_handler=rand_meas_handler,
        prefixB='test_circ',
    )

    # get circs
    circs = crossfid.bind_params_to_meas(point)
    assert len(circs) == num_random
    new_circs = fitter.bind_params_to_meas(point)
    assert len(new_circs) == 1

    # execute and test results processing
    results = exe_instance.execute(circs + new_circs)
    _ = fitter.evaluate_cost(results)
    assert fitter.calibrator.calibrated
    crossfid.comparison_results = results
    _ = crossfid.evaluate_cost(results)


def test_multi_strategy_fitter():
    """ """

    ansatz = RandomAnsatz(2, 2, strict_transpile=True)
    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    wpo = WeightedPauliOperator([
        (1., Pauli('ZZ')),
        (1., Pauli('XX')),
    ])
    cost = CostWPO(ansatz, instance, wpo)
    point = np.ones(ansatz.nb_params)

    strategies = [
        'zne,richardson,5',
        'zne,richardson,7',
        'pb,10,1000',
        'pb,20,1000',
        'pb,30,1000',
        'zne,linear,5',
        'none',
    ]
    strat_handler = MultiStrategyFitter(
        cost, strategies, pb_calibration_point=point)

    # type test on internal fitters
    assert isinstance(strat_handler.fitters[0], CXMultiplierFitter)
    assert isinstance(strat_handler.fitters[1], CXMultiplierFitter)
    assert isinstance(strat_handler.fitters[2], PurityBoostFitter)
    assert isinstance(strat_handler.fitters[3], PurityBoostFitter)
    assert isinstance(strat_handler.fitters[4], PurityBoostFitter)
    assert isinstance(strat_handler.fitters[5], CXMultiplierFitter)
    assert isinstance(strat_handler.fitters[6], CostWPO)

    # test number of circuits yielded
    circs = strat_handler.bind_params_to_meas(point)
    assert len(circs) == 2*3 + 2*4 + 2 + 2 + 2 + 30 + 2*3 + 2

    # test can evaluate
    results = instance.execute(circs)
    means, stds = strat_handler.evaluate_cost_and_std(results)
    assert len(means) == len(strategies)
    assert len(stds) == len(strategies)


def test_multi_strategy_fitter_reject_duplicates():
    """ """
    ansatz = RandomAnsatz(2, 2)
    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    wpo = WeightedPauliOperator([
        (1., Pauli('ZZ')),
        (1., Pauli('XX')),
    ])
    cost = CostWPO(ansatz, instance, wpo)

    strategies = [
        'zne,richardson,5',
        'zne,richardson,5',
    ]
    try:
        MultiStrategyFitter(cost, strategies)
        assert False, 'should have raised error'
    except ValueError:
        pass

