"""
Quantum error mitigation functions and classes.
"""

import abc

import numpy as np

from qiskit import QiskitError
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import random_unitary

from .cost import CostInterface, correlation_fixed_U, bind_params


def bootstrap_resample(stat_func, empirical_distribution, num_bootstraps,
                       return_dist=False):
    """
    Calculate the boostrap mean and standard-error of `stat_func` applied to
    `empirical_distribution` dataset. Optionally return the distribution of the
    resampled values of the estimator, instead of the standard error.

    Parameters
    ----------
    stat_func : Callable
        Function to evaluate on each bootstrap resample e.g. np.std
    empirical_distribution : np.ndarray
        Data to resample
    num_bootstraps : int
        Number of bootstrap resamples to perform
    return_dist : boolean, default False
        If True, return the bootstrapped distribution of the estimator instead
        of the standard error (see Returns)

    Returns
    -------
    mean_estimate : float
    standard_error OR estimator_dist : float OR list[float]
        If `return_dist=True` returns distribution, else returns standard error
    """
    num_data = empirical_distribution.size

    try:
        # try to vectorise the bootstrapping, unless the size of the resulting
        # array will be too large (134217728 is approx the size of a 1GB
        # float64 array)
        if num_bootstraps * num_data > 134217728:
            raise TypeError

        # vectorisation will also fail if `stat_func` does not have an `axis`
        # kwarg, which will raise a TypeError here
        resample_indexes = np.random.randint(
            0, num_data, size=(num_bootstraps, num_data))
        resampled_estimator = stat_func(
            empirical_distribution[resample_indexes], axis=1
            )
    except TypeError:
        # more memory safe but much slower
        resampled_estimator = np.zeros(num_bootstraps)
        for boot in range(num_bootstraps):
            resample_indexes = np.random.randint(0, num_data, size=num_data)
            resampled_estimator[boot] = stat_func(
                empirical_distribution[resample_indexes])

    if return_dist:
        # bootstrap estimate and distribution
        return (
            np.mean(resampled_estimator),
            resampled_estimator
        )
    # bootstrap estimate and standard deviation
    return (
        np.mean(resampled_estimator),
        (np.mean(resampled_estimator**2)
            - np.mean(resampled_estimator)**2)
    )


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


class FitterInterface(CostInterface, metaclass=abc.ABCMeta):
    """
    Interface for error mitigation fitters. Extends the CostInterface so that
    (if we feel comfortable black-boxing an error mitigation approach) Fitter
    objs could behave as Cost objs -- yielding all the necessary circuits to
    compute an error mitigated estimate of a cost and returning its value after
    execution on a backend.
    """


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
        return circuit
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


def richardson_extrapolation(cost_series, cost_series_vars=None, alphas=None):
    """
    see p22 of arXiv:2011.01382
    """

    # assume CX multiplication
    if alphas is None:
        alphas = list(range(1, len(cost_series), 2))

    betas = []
    for idx1, alpha1 in enumerate(alphas):
        betas[idx1] = 1.
        for idx2, alpha2 in enumerate(alphas):
            if not idx1 == idx2:
                betas[idx1] *= (alpha2 / (alpha1 - alpha2))

    variances = None
    if cost_series_vars is not None:
        variances = sum(
            [(beta**2) * var for beta, var in zip(betas, cost_series_vars)]
        )

    return (
        sum([beta*cost for beta, cost in zip(betas, cost_series)]),
        variances
    )


class ZNECXMultiplierFitter(FitterInterface):
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
        self.cost = cost_obj
        self.max_factor = max_factor
        self.extrapolation_strategy = extrapolation_strategy

    def bind_params_to_meas(self, params=None, params_names=None):
        """ """
        bound_circs = self.cost.bind_params_to_meas(
            params=params,
            params_names=params_names
        )

        zne_circs = []
        for factor in range(1, self.max_factor, 2):
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
        cost_series = [
            self.cost.evaluate_cost(results, name=name+'zne'+f'{idx}',
                                    **kwargs)
            for idx in range(1, self.max_factor, 2)
        ]

        raw_cost = self.cost.evaluate_cost(results, name=name, **kwargs)

        if not self.calibrator.calibrated:
            self.calibrator.process_calibration_results(results, name=name)

        mitigated_resamples = raw_cost / (1 - self.calibrator.ptot_dist)
        return np.mean(mitigated_resamples), np.std(mitigated_resamples)

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


def add_random_measurements(circuit, num_rand, seed=None):
    """
    Add single qubit measurements in Haar random basis to all the qubits, at
    the end of the circuit. Used to infer the purity of the output state.
    Copies the circuit so preserves registers, parameters and circuit name.
    Independent of what measurements were in the input circuit, all qubits will
    be measured.

    Parameters
    ----------
    circuit : qiskit circuit
    num_rand : int
        Number of random unitaries to use
    seed : int, optional
        Random number seed for reproducibility

    Returns
    -------
    purity_circuits : list of qiskit circuits
        Copies of input circuit(s), with random unitaries added to the end
    """
    rand_state = np.random.default_rng(seed)

    rand_meas_circuits = []
    for _ in range(num_rand):

        # copy circuit to preserve registers, but remove any final measurements
        new_circ = circuit.copy()
        new_circ.remove_final_measurements()
        # add random single qubit unitaries
        for qb_idx in range(new_circ.num_qubits):
            rand_gate = random_unitary(2, seed=rand_state)
            new_circ.append(rand_gate, [qb_idx])
        new_circ.measure_all()
        rand_meas_circuits.append(new_circ)

    return rand_meas_circuits


def estimate_purity_fixed_u(
    results,
    num_random=None,
    unitaries_set=None,
    names=str,
):
    """
    Extract the contributions towards the evaluation of the purity of a quantum
    state using random single qubit measurements (arxiv:1909.01282), resolved
    by each random unitary.

    Parameters
    ----------
    results :  qiskit.result.Result obj
        Results to calculate cross-fidelity between
    num_random : int, *optional*
    unitaries_set : list of ints, *optional*
        One of these two args must be supplied, with num_random taking
        precedence. Used to locate relevant measurement results in the
        qiksit result objs.
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

    # parse num_random/unitaries_set args
    if (num_random is None) and (unitaries_set is None):
        raise ValueError('Please specify either the number of random unitaries'
              + ' (`num_random`), or the specific indexes of the random'
              + ' unitaries to include (`unitaries_set`).')
    if num_random is not None:
        unitaries_set = range(num_random)
    else:
        num_random = len(unitaries_set)

    # iterate over the different random unitaries
    contributions_fixed_u = np.zeros(len(unitaries_set))
    for uidx in unitaries_set:

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

        contributions_fixed_u[uidx] = correlation_fixed_U(prob_rho_fixed_u,
                                                          prob_rho_fixed_u)

    # normalisation
    contributions_fixed_u = (2**nb_qubits)*contributions_fixed_u

    return contributions_fixed_u


def purity_from_random_measurements(
    results,
    num_random=None,
    unitaries_set=None,
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
    num_random : int, *optional*
    unitaries_set : list of ints, *optional*
        One of these two args must be supplied, with num_random taking
        precedence. Used to locate relevant measurement results in the
        qiksit result objs.
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
        results,
        num_random=num_random,
        unitaries_set=unitaries_set,
        names=names,
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
        num_random,
        seed=None,
        num_bootstraps=1000,
        calibration_point=None,
    ):
        """ """
        self.ansatz = ansatz
        self.instance = instance
        self.num_random = num_random
        self.seed = seed
        self.num_bootstraps = num_bootstraps

        if not calibration_point:
            # random vector with elements in [0,2\pi]
            self.calibration_point = np.random.random(size=len(ansatz.params))
            self.calibration_point = 2. * np.pi * self.calibration_point
        elif not calibration_point.size == len(ansatz.params):
            # test dim of calibration point
            raise ValueError('Dimensions of calibration_point do not match'
                             + ' number of params in ansatz.')
        else:
            self.calibration_point = calibration_point

        # have not yet estimated ptot
        self.ptot = None
        self.ptot_dist = None

        # function used to name circuits
        self._circ_name = lambda idx: 'ptot-cal-' + f'{idx}'

        # call base calibration init
        BaseCalibrator.__init__(self)

    def make_calibration_circuits(self):
        """ """
        bound_ansatz = bind_params(
            self.ansatz.circuit,
            self.calibration_point,
            self.ansatz.params
        )[0]
        calibration_circuits = add_random_measurements(
            bound_ansatz,
            self.num_random,
            seed=self.seed
        )

        # name circuits so they can be found
        for idx, circ in enumerate(calibration_circuits):
            circ.name = self._circ_name(idx)

        # transpile with instance ready for execution
        t_calibration_circuits = self.instance.transpile(calibration_circuits)

        return t_calibration_circuits

    def process_calibration_results(self, results, name=None):
        """ """
        if name is not None:
            def _circ_name(idx):
                return name + self._circ_name(idx)
        else:
            _circ_name = self._circ_name

        contributions_fixed_u = estimate_purity_fixed_u(
            results,
            num_random=self.num_random,
            names=_circ_name,
        )

        def compute_ptot(data, axis=0):
            n_qubits = self.ansatz.circuit.num_qubits
            purity = np.mean(data, axis=axis)
            return 1 - np.sqrt((2**n_qubits*purity - 1)/(2**n_qubits - 1))

        # bootstrap estimate of ptot and its distribution over the resamples
        self.ptot, self.ptot_dist = bootstrap_resample(
            compute_ptot,
            contributions_fixed_u,
            self.num_bootstraps,
            return_dist=True,
        )

        self.calibrated = True
        self._yielded_calibration_circuits = False

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


class PurityBoostFitter(FitterInterface):
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
        self.cost = cost_obj

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
        raw_cost = self.cost.evaluate_cost(results, name=name, **kwargs)

        if not self.calibrator.calibrated:
            self.calibrator.process_calibration_results(results, name=name)

        mitigated_resamples = raw_cost / (1 - self.calibrator.ptot_dist)
        return np.mean(mitigated_resamples), np.std(mitigated_resamples)

    def evaluate_cost(
        self,
        results,
        name='',
        **kwargs,
    ):
        """ """
        return self.evaluate_cost_and_std(results, name=name, **kwargs)[0]
