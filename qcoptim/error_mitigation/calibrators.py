"""
"""

import numpy as np

from ..cost.crossfidelity import _purity_per_u
from ..utilities import bootstrap_resample, RandomMeasurementHandler


class BaseCalibrator():
    """
    Base class for error mitigation calibration classes. Not all error
    mitigation strategies require calibration e.g. zero-noise extrapolation,
    but others do e.g. PurityBoosting and Clifford data regression. Calibrator
    classes black-box the calibration task. Calibrator instances can be shared
    between multiple fitters, in which case calibration is only performed once.
    """

    def __init__(self):
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


#
# Purity estimation
# -----------------
#


def purity_from_random_measurements(
    results,
    num_random,
    names=str,
    num_bootstraps=1000,
    vectorise=False,
    random_seed=None,
):
    """
    Function to calculate the purity of a quantum state using random single
    qubit measurements (arxiv:1909.01282). Assumes Haar random single qubit
    measurements have been added to the end of a circuit.

    Parameters
    ----------
    results : qiskit.result.Result
        Results to calculate cross-fidelity between
    num_random : int
        Number of random measurement basis used.
    names : Callable, optional
        Function that maps index of a random circuit to a name of a circuit in
        the qiskit results object
    num_bootstraps : int
        Number of bootstrap resamples to use in estimate of purity and std

    Returns
    -------
    float
        Mean of purity estimate
    float
        Standard error in estimate
    """
    contributions_fixed_u = _purity_per_u(
        results, num_random, names=names, vectorise=vectorise,
    )

    # bootstrap estimate and standard deviation
    return bootstrap_resample(
        np.mean, contributions_fixed_u, num_bootstraps,
        random_seed=random_seed
    )


class PurityBoostCalibrator(BaseCalibrator):
    """
    Calibration class for purity boost error-mitigation.
    """

    def __init__(
        self,
        ansatz,
        instance=None,
        num_random=None,
        seed=None,
        num_bootstraps=1000,
        calibration_point=None,
        circ_name=None,
        transpiler='instance',
        rand_meas_handler=None,
    ):
        """
        Parameters
        ----------
        ansatz : class implementing ansatz interface
            Ansatz obj
        instance : qiskit.utils.QuantumInstance, optional
            Quantum instance to use
        num_random : int
            Number of random basis to generate
        seed : int, optional
            Seed for generating random basis
        num_bootstraps : int, optional
            Number of bootstrap resamples used to estimate uncertainty in ptot
        calibration_point : None, optional
            Parameter point (w/r/t/ ansatz parameters) to calibrate at
        circ_name : callable, optional
            Function used to name circuits, should have signature `int -> str`
            and preferably should prefix the int, e.g. return something like
            'some-str'+str(int)`
        transpiler : str, optional
            Choose how to transpile circuits, current options are:
                'instance' : use quantum instance
                'pytket' : use pytket compiler
        rand_meas_handler : None, optional
            Can pass an already initialised RandomMeasurementHandler obj to use
            to generate random basis circuits internally. This can be shared
            with other users to avoid repeated random characterisation of the
            same state.

            Will raise ValueError if rand_meas_handler's ansatz or instance are
            different from the args, unless `ansatz=None` and `instance=None`.

            `rand_meas_handler.num_random` can be different from num_random as
            long as num_random is smaller.
        """
        self.num_bootstraps = num_bootstraps

        # set default for num_random if passed as None
        if num_random is None:
            if rand_meas_handler is None:
                num_random = 500
            else:
                num_random = rand_meas_handler.num_random

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

        # make internal RandomMeasurementHandler if none passed
        if rand_meas_handler is None:
            self._rand_meas_handler = RandomMeasurementHandler(
                ansatz,
                instance,
                num_random,
                seed=seed,
                circ_name=circ_name,
                transpiler=transpiler,
            )
        else:
            if ansatz is not None and ansatz != rand_meas_handler.ansatz:
                raise ValueError('Ansatz passed different from'
                                 + ' rand_meas_handler obj.')
            if instance is not None and instance != rand_meas_handler.instance:
                raise ValueError('Quantum instance passed different from'
                                 + ' rand_meas_handler obj.')
            if num_random > rand_meas_handler.num_random:
                raise ValueError('num_random larger than num_random of'
                                 + ' rand_meas_handler obj.')
            self._rand_meas_handler = rand_meas_handler
        self.num_random = num_random

        # call base calibration init
        BaseCalibrator.__init__(self)

    def make_calibration_circuits(self):
        """
        Generating circuits defered to a RandomMeasurementHandler in this class
        """
        return []

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
        Yield calibration circuits, using RandomMeasurementHandler's locking
        """
        return self._rand_meas_handler.circuits(self.calibration_point)

    def reset(self):
        """
        Lose memory of waiting for calibration results, and having completed
        calibration
        """
        self._rand_meas_handler.reset()
        self.calibrated = False

    def process_calibration_results(self, results, name=None, vectorise=False):
        """
        Parameters
        ----------
        results : qiskit.result.Result
            Results to use
        name : None, optional
            Extra prefix string to use to select from results
        """
        if name is not None:
            def _circ_name(idx):
                return name + self._rand_meas_handler.circ_name(idx)
        else:
            _circ_name = self._rand_meas_handler.circ_name

        contributions_fixed_u = _purity_per_u(
            results, self.num_random, names=_circ_name, vectorise=vectorise,
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
            random_seed=self.seed,
        )

        self._rand_meas_handler.reset()
        self.calibrated = True

    @property
    def ptot(self):
        return self._ptot

    @ptot.setter
    def ptot(self, ptot):
        if ptot is not None:
            if ptot < 0:
                self._ptot = 0.
            elif ptot > 1.:
                self._ptot = 1.
            else:
                self._ptot = ptot
        else:
            # allow setting to None
            self._ptot = None
