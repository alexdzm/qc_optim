"""
Quantum error mitigation functions and classes.
"""

# import abc

import numpy as np
from scipy.optimize import curve_fit

from qiskit import QiskitError
from qiskit.circuit.library import CXGate

from .cost import CostInterface
from .cost.crossfidelity import _correlation_fixed_u

from .utilities import bootstrap_resample, RandomMeasurementHandler


class BaseCalibrator():
    """
    Base class for error mitigation calibration classes. Not all error
    mitigation strategies require calibration e.g. zero-noise extrapolation,
    but others do e.g. PurityBoosting and Clifford data regression. Calibrator
    classes black-box the calibration task. Calibrator instances can be shared
    between multiple fitters, in which case calibration is only performed once.
    """

    def __init__(self, ):
        """ """
        self._calibration_circuits = self.make_calibration_circuits()
        self._yielded_calibration_circuits = False
        self.calibrated = False

    def make_calibration_circuits(self):
        """
        Implemented by derived classes
        """
        raise NotImplementedError

    @property
    def calibration_circuits(self):
        """
        Yield calibration circuits
        """
        if not self._yielded_calibration_circuits:
            self._yielded_calibration_circuits = True
            return self._calibration_circuits
        return []

    def reset(self):
        """
        Lose memory of waiting for calibration results, and having completed
        calibration
        """
        self._yielded_calibration_circuits = False
        self.calibrated = False

    def process_calibration_results(self, results):
        """
        Implemented by derived classes
        """
        raise NotImplementedError


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
    circuit : qiskit circuit
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
    see p22 of arXiv:2011.01382
    """
    betas = -1*np.ones(len(cost_series))
    for idx1, alpha1 in enumerate(stretch_factors):
        for idx2, alpha2 in enumerate(stretch_factors):
            if not idx1 == idx2:
                betas[idx1] *= (alpha2 / (alpha1 - alpha2))

    # runtime test of normalisation conditions on betas
    test_array = [
        sum([b*a**idx for (a, b) in zip(stretch_factors, betas)])
        for idx, _ in enumerate(stretch_factors)
    ]
    assert np.isclose(test_array[0], 1.)
    assert np.all(np.isclose(test_array[1:], np.zeros(len(test_array)-1)))

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
    """
    def _linear_fit(x, a, b):
        return a + b*x

    popt, pcov = curve_fit(_linear_fit, stretch_factors, cost_series,
                           sigma=np.sqrt(cost_series_vars))

    return popt[0], np.sqrt(pcov[0, 0])


class CXMultiplierFitter(BaseFitter):
    """
    """

    def __init__(
        self,
        cost_obj,
        max_factor,
        extrapolation_strategy='richardson',
    ):
        """
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
        """ """
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
        """ """
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
        """ """
        return self.evaluate_cost_and_std(results, name=name, **kwargs)[0]


#
# Purity estimation
# -----------------
#


def estimate_purity_fixed_u(results, num_random, names=str):
    """
    Extract the contributions towards the evaluation of the purity of a quantum
    state using random single qubit measurements (arxiv:1909.01282), resolved
    by each random unitary.

    Parameters
    ----------
    results :  qiskit.result.Result obj
        Results to calculate cross-fidelity between
    num_random : int
    names : Callable, optional
        Function that maps index of a random circuit to a name of a circuit in
        the qiskit results object

    Returns
    -------
    contributions_fixed_u : numpy.ndarray
        Contributions towards random measurement purity estimate, resolved for
        each single random measurement
    """
    nb_qubits = None

    # iterate over the different random unitaries
    contributions_fixed_u = np.zeros(num_random)
    for uidx in range(num_random):

        # try to extract matching experiment data
        try:
            countsdict_rho_fixed_u = results.get_counts(names(uidx))
        except QiskitError as missing_experiments:
            raise KeyError('Cannot extract matching experiment data to'
                           + ' calculate purity.') from missing_experiments

        # normalise counts dict to give empirical probability dists
        prob_rho_fixed_u = {k: v/sum(countsdict_rho_fixed_u.values())
                            for k, v in countsdict_rho_fixed_u.items()}

        # use this to check number of qubits has been consistent
        # over all random unitaries
        if nb_qubits is None:
            # get the first dict key string and find its length
            nb_qubits = len(list(prob_rho_fixed_u.keys())[0])
        if not nb_qubits == len(list(prob_rho_fixed_u.keys())[0]):
            raise ValueError(
                'nb_qubits=' + f'{nb_qubits}' + ', P_rhoA_fixed.keys()='
                + f'{prob_rho_fixed_u.keys()}'
            )

        contributions_fixed_u[uidx] = _correlation_fixed_u(prob_rho_fixed_u,
                                                           prob_rho_fixed_u)

    # normalisation
    contributions_fixed_u = (2**nb_qubits)*contributions_fixed_u

    return contributions_fixed_u


def purity_from_random_measurements(
    results,
    num_random,
    names=str,
    num_bootstraps=1000,
):
    """
    Function to calculate the purity of a quantum state using random single
    qubit measurements (arxiv:1909.01282). Assumes Haar random single qubit
    measurements have been added to the end of a circuit.

    Parameters
    ----------
    results :  qiskit.result.Result obj
        Results to calculate cross-fidelity between
    num_random : int.
    names : Callable, optional
        Function that maps index of a random circuit to a name of a circuit in
        the qiskit results object
    num_bootstraps : int
        Number of bootstrap resamples to use in estimate of purity and std

    Returns
    -------
    tr_rho_2 : float
    tr_rho_2_err : float
    """
    contributions_fixed_u = estimate_purity_fixed_u(
        results, num_random, names=names,
    )

    # bootstrap estimate and standard deviation
    return bootstrap_resample(np.mean, contributions_fixed_u, num_bootstraps)


class PurityBoostCalibrator(BaseCalibrator):
    """
    Calibration class for purity boost error-mitigation
    """

    def __init__(
        self,
        ansatz,
        instance,
        num_random=500,
        seed=None,
        num_bootstraps=1000,
        calibration_point=None,
        circ_name=None,
        rand_meas_handler=None,
    ):
        """ """
        self.num_bootstraps = num_bootstraps

        if calibration_point is None:
            # random vector with elements in [0,2\pi]
            self.calibration_point = np.random.random(size=len(ansatz.params))
            self.calibration_point = 2. * np.pi * self.calibration_point
        elif not calibration_point.size == len(ansatz.params):
            # test dim of calibration point
            raise ValueError('Dimensions of calibration_point do not match'
                             + ' number of params in ansatz.')
        else:
            self.calibration_point = calibration_point

        # this is to match others users of RandomMeasurementHandler,
        # e.g. CrossFidelity class, which assume 2d parameter points
        self.calibration_point = np.atleast_2d(self.calibration_point)

        # have not yet estimated ptot
        self.ptot = None
        self.ptot_std = None

        # default circ names
        if circ_name is None:
            def circ_name(idx):
                return 'ptot-cal'+f'{idx}'

        # make internal RandomMeasurementHandler in none passed
        if rand_meas_handler is None:
            self._rand_meas_handler = RandomMeasurementHandler(
                ansatz,
                instance,
                num_random,
                seed=seed,
                circ_name=circ_name,
            )
        else:
            self._rand_meas_handler = rand_meas_handler

        # call base calibration init
        BaseCalibrator.__init__(self)

    def make_calibration_circuits(self):
        """ """
        return []

    @property
    def num_random(self):
        return self._rand_meas_handler.num_random

    @property
    def seed(self):
        return self._rand_meas_handler.seed

    @property
    def ansatz(self):
        return self._rand_meas_handler.ansatz

    @property
    def instance(self):
        return self._rand_meas_handler.instance

    @property
    def calibration_circuits(self):
        """
        Yield calibration circuits
        """
        return self._rand_meas_handler.circuits(self.calibration_point)

    def reset(self):
        """
        Lose memory of waiting for calibration results, and having completed
        calibration
        """
        self._rand_meas_handler.reset()
        self.calibrated = False

    def process_calibration_results(self, results, name=None):
        """ """
        if name is not None:
            def _circ_name(idx):
                return name + self._rand_meas_handler.circ_name(idx)
        else:
            _circ_name = self._rand_meas_handler._circ_name

        contributions_fixed_u = estimate_purity_fixed_u(
            results, self.num_random, names=_circ_name,
        )

        def compute_ptot(data, axis=0):
            n_qubits = self.ansatz.circuit.num_qubits
            purity = np.mean(data, axis=axis)
            return 1 - np.sqrt((2**n_qubits*purity - 1)/(2**n_qubits - 1))

        # bootstrap estimate of ptot and its distribution over the resamples
        self.ptot, self.ptot_std = bootstrap_resample(
            compute_ptot,
            contributions_fixed_u,
            self.num_bootstraps,
        )

        self._rand_meas_handler.reset()
        self.calibrated = True

    @property
    def ptot(self):
        """ Getter for ptot attribute """
        return self._ptot

    @ptot.setter
    def ptot(self, ptot):
        """ Setter for ptot attribute """
        if ptot:
            if ptot < 0:
                self._ptot = 0.
            elif ptot > 1.:
                self._ptot = 1.
            else:
                self._ptot = ptot
        else:
            # allow setting to None
            self._ptot = None


class PurityBoostFitter(BaseFitter):
    """
    arXiv:2101.01690

    Currently only works for cost objects corresponding to the expectation
    values of traceless operators.
    """

    def __init__(
        self,
        cost_obj,
        calibrator=None,
        num_random=500,
        seed=None,
        num_bootstraps=1000,
        calibration_point=None,
    ):
        """
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
            self.calibrator = calibrator

    def bind_params_to_meas(self, params=None, params_names=None):
        """ """
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
        **kwargs,
    ):
        """ """
        std_func = getattr(self.cost, "evaluate_cost_and_std", None)
        if callable(std_func):
            raw_cost, raw_std = std_func(results, name=name, **kwargs)
        else:
            raw_std = None
            raw_cost = self.cost.evaluate_cost(results, name=name, **kwargs)

        if not self.calibrator.calibrated:
            self.calibrator.process_calibration_results(results, name=name)

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
        }

        return mean, np.sqrt(var)

    def evaluate_cost(
        self,
        results,
        name='',
        **kwargs,
    ):
        """ """
        return self.evaluate_cost_and_std(results, name=name, **kwargs)[0]
