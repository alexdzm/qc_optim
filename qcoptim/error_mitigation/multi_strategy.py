"""
"""

import numpy as np

from ..utilities import RandomMeasurementHandler

from .fitters import (
    BaseFitter,
    PurityBoostFitter,
    CXMultiplierFitter,
)
from .calibrators import PurityBoostCalibrator


_CALIBRATION_CIRCUIT_PREFIX = 'calibration_circ'


def _handle_purity_boosting_strategies(
    cost_obj,
    pb_strategies,
    seed=None,
    calibration_point=None,
    transpiler='instance',
):
    """
    """
    # identify largest num_random in purity-boosts
    arg_largest_pb = max(range(len(pb_strategies)),
                         key=lambda i: int(pb_strategies[i][1]))

    # make shared RandomMeasurementHandler with largest num_random
    def circ_name(idx):
        return _CALIBRATION_CIRCUIT_PREFIX + f'{idx}'
    rand_meas_handler = RandomMeasurementHandler(
        cost_obj.ansatz,
        cost_obj.instance,
        int(pb_strategies[arg_largest_pb][1]),
        transpiler=transpiler,
        seed=seed,
        circ_name=circ_name,
    )

    # all of the calibrators need to have the same calibration point
    if calibration_point is None:
        calibration_point = np.random.random(size=len(cost_obj.ansatz.params))
        calibration_point = 2. * np.pi * calibration_point

    # make calibrators that are potentially subsamples of rand_meas_handler and
    # feed these directly into PurityBoostFitter
    fitters = {}
    for strat in pb_strategies:
        calibrator = PurityBoostCalibrator(
            cost_obj.ansatz,
            num_random=int(strat[1]),
            rand_meas_handler=rand_meas_handler,
            num_bootstraps=int(strat[2]),
            calibration_point=calibration_point,
        )
        fitters[','.join(strat)] = PurityBoostFitter(
            cost_obj, calibrator=calibrator,
        )

    return fitters


def _handle_zne_strategies(cost_obj, zne_strategies):
    """ """
    fitters = {}
    for strat in zne_strategies:
        fitters[','.join(strat)] = CXMultiplierFitter(
            cost_obj, int(strat[2]), extrapolation_strategy=strat[1]
        )
    return fitters


def _parse_error_mitigation_strategies(
    cost_obj,
    error_mitigation_strategies,
    pb_calibration_point=None,
    transpiler='instance',
):
    """
    """
    # split up ZNE and purity-boost
    _strategies = [strat.split(',') for strat in error_mitigation_strategies]
    zne_strategies = [strat for strat in _strategies if strat[0] == 'zne']
    pb_strategies = [strat for strat in _strategies if strat[0] == 'pb']

    # handle purity-boosting
    if len(pb_strategies) > 0:
        pb_fitters = _handle_purity_boosting_strategies(
            cost_obj, pb_strategies, transpiler=transpiler,
            calibration_point=pb_calibration_point)

    # handle zne
    if len(zne_strategies) > 0:
        zne_fitters = _handle_zne_strategies(cost_obj, zne_strategies)

    # order to match input
    fitters = []
    for strat in _strategies:
        if strat[0] == 'zne':
            fitters.append(zne_fitters[','.join(strat)])
        elif strat[0] == 'pb':
            fitters.append(pb_fitters[','.join(strat)])
        elif strat[0] == 'none':
            # allow a do-nothing fitter
            fitters.append(cost_obj)
        else:
            raise ValueError('Unrecognized error mitigation strategy: '
                             + f'{strat[0]}')

    return fitters


class MultiStrategyFitter(BaseFitter):
    """ """

    def __init__(
        self,
        cost_obj,
        error_mitigation_strategies,
        pb_calibration_point=None,
        transpiler='instance',
    ):
        """ """
        BaseFitter.__init__(self, cost_obj)

        # test for duplicates and raise error
        if any(
            error_mitigation_strategies.count(element) > 1
            for element in error_mitigation_strategies
        ):
            raise ValueError('Strategy list contains duplicates.')

        self.strategies = error_mitigation_strategies
        self.fitters = _parse_error_mitigation_strategies(
            cost_obj, error_mitigation_strategies, transpiler=transpiler,
            pb_calibration_point=pb_calibration_point)

    def bind_params_to_meas(self, params=None, params_names=None):
        """ """
        bound_circs = []
        for strat, fitter in zip(self.strategies, self.fitters):
            tmp = fitter.bind_params_to_meas(
                params=params, params_names=params_names)

            for circ in tmp:
                # prefix circuit names, except calibration circuits
                if _CALIBRATION_CIRCUIT_PREFIX not in circ.name:
                    circ.name = strat + '-' + circ.name

                bound_circs.append(circ)

        return bound_circs

    def evaluate_cost_and_std(self, results, name='', **kwargs, ):
        """ """
        self.last_evaluation = {}
        means, stds = [], []
        for strat, fitter in zip(self.strategies, self.fitters):
            mean, std = fitter.evaluate_cost_and_std(
                results, name=name+strat+'-', calibration_name=name, **kwargs)
            means.append(mean)
            stds.append(std)
            self.last_evaluation[strat] = fitter.last_evaluation
        return means, stds

    def evaluate_cost(self, results, name='', **kwargs, ):
        """ """
        means, _ = self.evaluate_cost_and_std(results, name=name, **kwargs)
        return means
