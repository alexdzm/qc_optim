"""
"""

import numpy as np
from scipy import linalg

from qiskit.circuit.library import CXGate

from ..cost import CostInterface

from .calibrators import PurityBoostCalibrator


class BaseFitter(CostInterface):
    """
    Base class for error mitigation fitters. Extends the CostInterface so that
    (if we feel comfortable black-boxing an error mitigation approach) Fitter
    objs could behave as Cost objs -- yielding all the necessary circuits to
    compute an error mitigated estimate of a cost and returning its value after
    execution on a backend.
    """
    def __init__(self, cost_obj):
        """
        Parameters
        ----------
        cost_obj : class implementing CostInterface
            Cost obj to error mitigate for
        """
        self.cost = cost_obj

        # for saving last evaluation
        self.last_evaluation = None


#
# Zero-noise extrapolation
# ------------------------
#


def multiply_cx(circuit, multiplier):
    """
    A simple implementation of zero-noise extrapolation increases the level of
    noise by some factor `multipler` by adding redundant CNOT's (the main
    source of circuit noise). Copies the circuit so preserves registers,
    parameters and circuit name. Will also match the measurements in the input
    circuit.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Description
    multiplier : int
        Multiplication factor for CNOT gates

    Returns
    -------
    multiplied_circuit : qiskit circuit
        Copy of input circuit, with CNOTs multiplied by multiplier
    """
    if multiplier == 1:
        return circuit.copy()
    if (not multiplier % 2 == 1) and (not isinstance(multiplier, int)):
        raise ValueError('multiplier must be an odd integer, recieved: '
                         + f'{multiplier}')

    # copy circuit to preserve registers
    multiplied_circuit = circuit.copy()
    # wipe instructions in copy
    multiplied_circuit.data = []

    # iterate over circuit instructions, adding them back to the copy but
    # multiplying CNOT's
    for instuction in circuit.data:
        multiplied_circuit.append(*instuction)
        if isinstance(instuction[0], CXGate):
            for _ in range(multiplier-1):
                multiplied_circuit.append(*instuction)

    return multiplied_circuit


def richardson_extrapolation(stretch_factors, cost_series,
                             cost_series_vars=None):
    """
    Apply Richardson extrapolation to obtain zero-noise estimate using
    `cost_series` measurements (with variances `cost_series_vars`) at
    `stretch_factors` noise amplifications. See e.g. p22 of arXiv:2011.01382

    Parameters
    ----------
    stretch_factors : list, numpy.ndarray
        Noise amplification factors
    cost_series : list, numpy.ndarray
        Measurements at different noise amplifications
    cost_series_vars : list, numpy.ndarray, optional
        Variances in measurements

    Returns
    -------
    float
        Zero-noise extrapolated mean estimate
    float
        Variance in zero-noise extrapolation (None if `cost_series_vars` not
        supplied)
    """
    betas = np.ones(len(cost_series))
    for idx1, alpha1 in enumerate(stretch_factors):
        for idx2, alpha2 in enumerate(stretch_factors):
            if not idx1 == idx2:
                betas[idx1] *= (alpha2 / (alpha1 - alpha2))

    # runtime test of normalisation conditions on betas
    test_array = [
        sum([b*a**idx for (a, b) in zip(stretch_factors, betas)])
        for idx, _ in enumerate(stretch_factors)
    ]
    if not (
        np.all(np.isclose(test_array[1:], np.zeros(len(test_array)-1)))
        or np.isclose(np.abs(test_array[0]), 1.)
    ):
        raise ValueError('Problem with Richardson coefficiants.')
    if test_array[0] < 0:
        betas = -1*betas

    # get std_err if possible
    std_err = None
    if cost_series_vars is not None:
        std_err = np.sqrt(sum(
            [(beta**2) * var for beta, var in zip(betas, cost_series_vars)]
        ))

    return (
        sum([beta*cost for beta, cost in zip(betas, cost_series)]),
        std_err
    )


def linear_extrapolation(stretch_factors, cost_series,
                         cost_series_vars=None):
    """
    Use linear fit to extrpolate to zero-noise using `cost_series` measurements
    (with variances `cost_series_vars`) at `stretch_factors` noise
    amplifications.

    Parameters
    ----------
    stretch_factors : list, numpy.ndarray
        Noise amplification factors
    cost_series : list, numpy.ndarray
        Measurements at different noise amplifications
    cost_series_vars : list, numpy.ndarray, optional
        Variances in measurements

    Returns
    -------
    float
        Zero-noise extrapolated mean estimate
    float
        Variance in zero-noise extrapolation
    """

    # want to ensure that the variances aren't too different, use the condition
    # that at most one of them can be more than 10% away from the mean. If this
    # isn't obeyed we will use the max variance in the final error estimate,
    # else use the mean
    _spread = np.abs(
        (cost_series_vars - np.mean(cost_series_vars))
        / np.mean(cost_series_vars)
    )
    if np.count_nonzero(_spread > 0.1) > 1:
        sigma_squared = np.max(cost_series_vars)
    else:
        sigma_squared = np.mean(cost_series_vars)

    # in this simple case do the linear regression by hand
    Xmat = np.array([np.ones(len(stretch_factors)), stretch_factors]).T
    inv_XtX = linalg.inv(Xmat.T @ Xmat)

    betas = inv_XtX @ Xmat.T @ np.array([cost_series]).T
    var_betas = inv_XtX * sigma_squared

    return betas[0], np.sqrt(var_betas[0, 0])


class CXMultiplierFitter(BaseFitter):
    """
    Error mitigation fitter that uses CX multiplication to attempt zero-noise
    extrapolation.
    """

    def __init__(
        self,
        cost_obj,
        max_factor,
        extrapolation_strategy='richardson',
    ):
        """
        Parameters
        ----------
        cost_obj : class implenting CostInterface
            Cost obj to apply error mitigation to
        max_factor : int
            Largest CX multiplication to go up to, must be odd
        extrapolation_strategy : str, optional
            Zero-noise extrapolation approach, supported:
                'richardson' : Richardson extrapolation
                'linear' : Use linear fit
        """
        BaseFitter.__init__(self, cost_obj)

        self.max_factor = max_factor
        self.stretch_factors = list(range(1, self.max_factor+1, 2))
        self.extrapolation_strategy = extrapolation_strategy

        if extrapolation_strategy == 'richardson':
            def _richardson_extrapolation(cost_series, cost_series_vars):
                return richardson_extrapolation(
                    stretch_factors=self.stretch_factors,
                    cost_series=cost_series,
                    cost_series_vars=cost_series_vars,
                )
            self.extrapolator = _richardson_extrapolation
        elif extrapolation_strategy == 'linear':
            def _linear_extrapolation(cost_series, cost_series_vars):
                return linear_extrapolation(
                    stretch_factors=self.stretch_factors,
                    cost_series=cost_series,
                    cost_series_vars=cost_series_vars,
                )
            self.extrapolator = _linear_extrapolation
        else:
            raise ValueError(extrapolation_strategy)

    def bind_params_to_meas(self, params=None, params_names=None):
        """
        Return measurement circuits bound at `params`

        Parameters
        ----------
        params : numpy.ndarray, optional
            Point (1d) or points (2d) to bind circuits at
        params_names : None, optional
            Description

        Returns
        -------
        list[qiskit.QuantumCircuit]
            Evaluation circuits
        """
        bound_circs = self.cost.bind_params_to_meas(
            params=params,
            params_names=params_names
        )

        zne_circs = []
        for factor in self.stretch_factors:
            for circ in bound_circs:
                tmp = multiply_cx(circ, factor)
                tmp.name = 'zne' + f'{factor}' + tmp.name
                zne_circs.append(tmp)

        return zne_circs

    def evaluate_cost_and_std(
        self,
        results,
        name='',
        **kwargs,
    ):
        """
        Evaluate cost and variance, and apply zero-noise extrapolation to them.

        Parameters
        ----------
        results : qiskit.result.Result
            Qiskit results obj
        name : str, optional
            Prefix on results names to find results data

        Returns
        -------
        float
            Zero-noise extrapolated mean estimate
        float
            Standard deviation in estimate
        """
        # see if cost obj has evaluate_cost_and_std method
        std_func = getattr(self.cost, "evaluate_cost_and_std", None)
        if callable(std_func):
            raw_cost_series = []
            raw_cost_vars = []
            for factor in self.stretch_factors:
                mean, std = std_func(results, name=name+'zne'+f'{factor}',
                                     **kwargs)
                raw_cost_series.append(mean)
                raw_cost_vars.append(std**2)
        else:
            raw_cost_vars = None
            raw_cost_series = [
                self.cost.evaluate_cost(results, name=name+'zne'+f'{idx}',
                                        **kwargs)
                for idx in self.stretch_factors
            ]

        # extrapolate to zero
        mean, std = self.extrapolator(raw_cost_series, raw_cost_vars)

        # save last evaluation
        self.last_evaluation = {
            'stretch_factors': self.stretch_factors,
            'raw_cost_series': raw_cost_series,
            'raw_cost_vars': raw_cost_vars,
            'zne_mean': mean,
            'zne_std': std,
        }

        return mean, std

    def evaluate_cost(
        self,
        results,
        name='',
        **kwargs,
    ):
        """
        Evaluate cost, and apply zero-noise extrapolation.

        Parameters
        ----------
        results : qiskit.result.Result
            Qiskit results obj
        name : str, optional
            Prefix on results names to find results data

        Returns
        -------
        float
            Zero-noise extrapolated mean estimate
        """
        return self.evaluate_cost_and_std(results, name=name, **kwargs)[0]


#
# Purity estimation
# -----------------
#


class PurityBoostFitter(BaseFitter):
    """
    Error mitigation fitter that uses a purity boosting technique
    (arXiv:2101.01690).

    ***Currently only works for cost objects corresponding to the expectation
    values of traceless operators.***
    """

    def __init__(
        self,
        cost_obj,
        num_random=500,
        seed=None,
        num_bootstraps=1000,
        calibration_point=None,
        calibrator=None,
    ):
        """
        Parameters
        ----------
        cost_obj : class implenting CostInterface
            Cost obj to apply error mitigation to
        num_random : int
            Number of random basis to generate
        seed : int, optional
            Seed for generating random basis
        num_bootstraps : int, optional
            Number of bootstrap resamples used to estimate uncertainty
        calibration_point : None, optional
            Parameter point (w/r/t/ ansatz parameters) to calibrate at
        calibrator : None, optional
            Can pass an already initialised Calibrator obj to use. This
            calibrator could have already been calibrated. Or, it may shared
            with other users to avoid repeated random characterisation of the
            same state.
            Will raise ValueError if calibrator's ansatz or instance are
            different from the cost_obj's.
        """

        BaseFitter.__init__(self, cost_obj)

        if calibrator is None:
            # make calibrator if none passed
            self.calibrator = PurityBoostCalibrator(
                ansatz=self.cost.ansatz,
                instance=self.cost.instance,
                num_random=num_random,
                seed=seed,
                num_bootstraps=num_bootstraps,
                calibration_point=calibration_point,
            )
        else:
            if self.cost.ansatz != calibrator.ansatz:
                raise ValueError('Cost and calibrator have different ansatz.')
            if self.cost.instance != calibrator.instance:
                raise ValueError('Cost and calibrator have different quantum'
                                 + ' instance.')
            self.calibrator = calibrator

    def bind_params_to_meas(self, params=None, params_names=None):
        """
        Return measurement circuits bound at `params`

        Parameters
        ----------
        params : numpy.ndarray, optional
            Point (1d) or points (2d) to bind circuits at
        params_names : None, optional
            Description

        Returns
        -------
        list[qiskit.QuantumCircuit]
            Evaluation circuits
        """
        bound_circs = self.cost.bind_params_to_meas(
            params=params,
            params_names=params_names
        )

        if not self.calibrator.calibrated:
            bound_circs += self.calibrator.calibration_circuits

        return bound_circs

    def evaluate_cost_and_std(
        self,
        results,
        name='',
        calibration_name=None,
        **kwargs,
    ):
        """
        Evaluate cost and variance, and apply purity boosting to them.

        Parameters
        ----------
        results : qiskit.result.Result
            Qiskit results obj
        name : str, optional
            Prefix on results names to find results data

        Returns
        -------
        float
            Purity boosted mean estimate
        float
            Standard deviation in estimate
        """
        if calibration_name is None:
            calibration_name = name

        std_func = getattr(self.cost, "evaluate_cost_and_std", None)
        if callable(std_func):
            raw_cost, raw_std = std_func(results, name=name, **kwargs)
        else:
            raw_std = None
            raw_cost = self.cost.evaluate_cost(results, name=name, **kwargs)

        if not self.calibrator.calibrated:
            self.calibrator.process_calibration_results(
                results, name=calibration_name)

        mean = raw_cost / (1 - self.calibrator.ptot)
        var = (
            (mean**2) * (self.calibrator.ptot_std**2
                         / (1 - self.calibrator.ptot)**2)
        )
        if raw_std is not None:
            # propagate error from uncertainty of cost estimate if available
            var += (mean**2) * (raw_std**2 / raw_cost**2)

        # store details of last evaluation
        self.last_evaluation = {
            'raw_cost': raw_cost,
            'raw_std': raw_std,
            'pboost_mean': mean,
            'pboost_std': np.sqrt(var),
            'ptot': self.calibrator.ptot,
            'ptot_std': self.calibrator.ptot_std,
        }

        return mean, np.sqrt(var)

    def evaluate_cost(
        self,
        results,
        name='',
        calibration_name=None,
        **kwargs,
    ):
        """
        Evaluate cost, and apply purity boosting.

        Parameters
        ----------
        results : qiskit.result.Result
            Qiskit results obj
        name : str, optional
            Prefix on results names to find results data

        Returns
        -------
        float
            Purity boosted mean estimate
        """
        return self.evaluate_cost_and_std(
            results, name=name, calibration_name=calibration_name, **kwargs
        )[0]
