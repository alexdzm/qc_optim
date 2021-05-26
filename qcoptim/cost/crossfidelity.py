"""
Cross-fidelity cost classes and functions
"""

import warnings
import functools

import numpy as np
import scipy as sp

from qiskit import QiskitError
from qiskit.result import Result

from ..utilities import (
    bootstrap_resample,
    RandomMeasurementHandler,
)

from .core import CostInterface


class CrossFidelity(CostInterface):
    """
    Cost class to implement offline CrossFidelity measurements between two
    quantum states (arxiv:1909.01282).
    """
    def __init__(
        self,
        ansatz,
        instance=None,
        nb_random=None,
        seed=None,
        comparison_results=None,
        num_bootstraps=1000,
        prefixA='CrossFid',
        prefixB='CrossFid',
        transpiler='instance',
        rand_meas_handler=None,
    ):
        """
        Parameters
        ----------
        ansatz : object implementing AnsatzInterface
            The ansatz object that this cost can be optimsed over
        instance : qiskit.utils.QuantumInstance
            Will be used to generate internal transpiled circuits
        nb_random : int, optional
            The number of random unitaries to average over
        seed : int, optional
            Seed used to generate random unitaries
        comparison_results : {dict, None, qiskit.results.Result}
            The use cases where None would be passed is if we are using
            this object to generate the comparison_results object for a
            future instance of CrossFidelity. This robustly ensures that
            the results objs to compare are compatible.
            If dict is passed it should be a qiskit results object that
            has been converted to a dict using its `to_dict` method.
            Ideally this would have been tagged with CrossFidelity
            metadata using this classes `tag_results_metadata` method.
            Can also accept a qiskit results obj
        num_bootstraps : int, optional
            Number of bootstrap resamples to use to estimate the standard error
            w/r/t/ the random unitaries
        prefixA : str, optional
            Prefix string to use on circuits generate to characterise A system.
        prefixB : str, optional
            Prefix string to use to extract system B's results from
            comparison_results.
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

            `rand_meas_handler.num_random` can be different from nb_random as
            long as nb_random is smaller.
        """
        self._prefixB = prefixB
        self._num_bootstraps = num_bootstraps

        # set default for nb_random if passed as None
        if nb_random is None:
            if rand_meas_handler is None:
                nb_random = 5
            else:
                nb_random = rand_meas_handler.num_random
        elif (not isinstance(nb_random, int)) or (nb_random < 1):
            raise ValueError('nb_random is invalid.')

        # make internal RandomMeasurementHandler in none passed
        def circ_name(idx):
            return prefixA + f'{idx}'
        if rand_meas_handler is None:
            self._rand_meas_handler = RandomMeasurementHandler(
                ansatz,
                instance,
                nb_random,
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
            if nb_random > rand_meas_handler.num_random:
                raise ValueError('nb_random larger than num_random of'
                                 + ' rand_meas_handler obj.')
            self._rand_meas_handler = rand_meas_handler
        self.nb_random = nb_random

        # run setter (see below)
        self.comparison_results = comparison_results

        # related to last evaluation
        self.last_evaluation = None

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
    def comparison_results(self):
        return self._comparison_results

    @comparison_results.setter
    def comparison_results(self, results):
        """
        Setter for comparison_results, perform validations
        """
        if results is not None:

            if (
                isinstance(results, dict)
                and 'cross-fidelity' in results.keys()
            ):

                # compare metadata
                comparison_metadata = results['crossfidelity_metadata']
                if self.seed != comparison_metadata['seed']:
                    raise ValueError(
                        'Comparison results use different random seed.'
                    )
                if self.nb_random > comparison_metadata['nb_random']:
                    raise ValueError(
                        'Comparison results use fewer random basis.'
                    )
                if self._prefixB != comparison_metadata['prefixA']:
                    raise ValueError(
                        'Passed prefixB does not match prefixA of comparison'
                        + ' results.'
                    )

                # bug fix, need counts dict keys to be hex values
                for val in results['results']:
                    new_counts = {}
                    for ckey, cval in val['data']['counts'].items():
                        # detect it is not hex
                        if not ckey[:2] == '0x':
                            ckey = hex(int(ckey, 2))
                        new_counts[ckey] = cval
                    val['data']['counts'] = new_counts

            else:
                warnings.warn(
                    'Warning, input results is not a dictionary containing'
                    + ' crossfidelity_metadata and so we cannot confirm that'
                    + ' the results are compatible. If the input results'
                    + ' object was collecting by this class consider using'
                    + ' the tag_results_metadata method to add the'
                    + ' crossfidelity_metadata.'
                )

            if isinstance(results, Result):
                self._comparison_results = results
            elif isinstance(results, dict):
                self._comparison_results = Result.from_dict(results)
            else:
                raise TypeError(
                    'comparison_results recieved type '+f'{type(results)}'
                )

    def tag_results_metadata(self, results):
        """
        Adds in CrossFidelity metadata to a results object. This can be
        used to ensure that two results sets are compatible.

        Parameters
        ----------
        results : qiskit.result.Result, dict
            The results data to process

        Returns
        -------
        results : dict
            Results dictionary with the CrossFidelity metadata added
        """

        # convert results to dict if needed
        if not isinstance(results, dict):
            results = results.to_dict()
        # add CrossFidelity metadata
        results.update({
            'crossfidelity_metadata': {
                'seed': self.seed,
                'nb_random': self.nb_random,
                'prefixA': self._rand_meas_handler.circ_name(''),
                }
            })
        return results

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
        if params is None:
            raise ValueError('Bound circuits requested without params given.')

        params = np.atleast_2d(params)
        if isinstance(params_names, str):
            params_names = [params_names]
        if params_names is None:
            params_names = [None] * len(params)
        elif not len(params_names) == len(params):
            raise ValueError(
                'params_names passed has different lengh to params.'
            )

        # get circuits from rand_meas_handler
        return self._rand_meas_handler.circuits(params)

    def evaluate_cost(
        self,
        results,
        name='',
        vectorise=True,
        **kwargs
    ):
        """
        Calculates the cross-fidelity using two sets of qiskit results.
        The variable names are chosen to match arxiv:1909.01282 as close
        as possible.

        Parameters
        ----------
        results : qiskit.result.Result
            Results to calculate cross-fidelity with, against the stored
            results dictionary.
        name : str, optional
            Prefix on results names to find results data
        **kwargs
            Description

        Returns
        -------
        float
            Evaluated cross-fidelity
        """
        return self.evaluate_cost_and_std(
            results, name=name, vectorise=vectorise, **kwargs)[0]

    def evaluate_cost_and_std(
        self,
        results,
        name='',
        vectorise=True,
        **kwargs
    ):
        """
        Calculates the cross-fidelity using two sets of qiskit results.
        The variable names are chosen to match arxiv:1909.01282 as close
        as possible.

        Parameters
        ----------
        results : Qiskit results type
            Results to calculate cross-fidelity with, against the stored
            results dictionary.
        name : str, optional
            Prefix on results names to find results data
        **kwargs
            Description

        Returns
        -------
        mean : float
            Evaluated cross-fidelity
        std : float
            Standard error on cross-fidelity estimation, obtained from
            bootstrap resampling
        """

        # we make it possible to instance a CrossFidelity obj without a
        # comparison_results dict so that we can easily generate the
        # comparison data using the same setup (e.g. seed, prefix). But
        # in that case cannote evaluate the cost.
        if self._comparison_results is None:
            raise ValueError('No comparison results set has been passed to'
                             + ' CrossFidelity obj.')

        # circuit naming functions
        def circ_namesA(idx):
            return name + self._rand_meas_handler.circ_name(idx)
        def circ_namesB(idx):
            return self._prefixB + f'{idx}'

        (dist_tr_rhoA_rhoB,
         dist_tr_rhoA_2,
         dist_tr_rhoB_2) = _crossfidelity_fixed_u(
         results, self._comparison_results, self.nb_random,
         circ_namesA=circ_namesA, circ_namesB=circ_namesB,
         vectorise=vectorise,
        )

        # bootstrap resample for means and std-errs
        tr_rhoA_rhoB, tr_rhoA_rhoB_err = bootstrap_resample(
            np.mean, dist_tr_rhoA_rhoB, self._num_bootstraps,
            random_seed=self.seed,
        )
        tr_rhoA_2, tr_rhoA_2_err = bootstrap_resample(
            np.mean, dist_tr_rhoA_2, self._num_bootstraps,
            random_seed=self.seed,
        )
        tr_rhoB_2, tr_rhoB_2_err = bootstrap_resample(
            np.mean, dist_tr_rhoB_2, self._num_bootstraps,
            random_seed=self.seed,
        )

        # divide by largest
        if tr_rhoA_2 > tr_rhoB_2:
            mean = tr_rhoA_rhoB / tr_rhoA_2
            std = (mean**2) * (
                (tr_rhoA_rhoB_err**2) / (tr_rhoA_rhoB)**2
                + (tr_rhoA_2_err)**2 / (tr_rhoA_2)**2
            )
        else:
            mean = tr_rhoA_rhoB / tr_rhoB_2
            std = (mean**2) * (
                (tr_rhoA_rhoB_err**2) / (tr_rhoA_rhoB)**2
                + (tr_rhoB_2_err)**2 / (tr_rhoB_2)**2
            )

        # store results of evaluation
        self.last_evaluation = {
            'tr_rhoA_rhoB': tr_rhoA_rhoB,
            'tr_rhoA_rhoB_err': tr_rhoA_rhoB_err,
            'tr_rhoA_2': tr_rhoA_2,
            'tr_rhoA_2_err': tr_rhoA_2_err,
            'tr_rhoB_2': tr_rhoB_2,
            'tr_rhoB_2_err': tr_rhoB_2_err,
            'mean': mean,
            'std': std,
        }

        return mean, std


def _unpack_experiment(idx, results):
    """
    Parameters
    ----------
    idx : int
        Index of experiment to access
    results : qiskit.result.Result
        Results object

    Returns
    -------
    num_qubits : int
        Number of qubits measured in experiment
    hexstrings : list[str]
        Keys of the counts, given as hexidecimal strings
    counts : numpy.ndarray
        Numpy array with the measurement counts
    """
    experiment = results.results[idx]
    num_qubits = experiment.header.n_qubits
    hexstrings = list(experiment.data.counts.keys())
    counts = np.array(list(experiment.data.counts.values()))
    return num_qubits, hexstrings, counts


def _crossfidelity_fixed_u(
    resultsA,
    resultsB,
    nb_random,
    circ_namesA=None,
    circ_namesB=None,
    vectorise=False,
):
    """
    Function to calculate the offline CrossFidelity between two quantum states
    (arxiv:1909.01282).

    Parameters
    ----------
    resultsA : qiskit.result.Result
        Results for system A
    resultsB : qiskit.result.Result
        Results for system B
    nb_random : int
        Number of random measurement basis
    circ_namesA : callable, optional
        Naming function for A results, `int -> str`
    circ_namesB : callable, optional
        Naming function for B results, `int -> str`

    Returns
    -------
    float
        tr_rhoA_rhoB, unnormalised cross-fidelity
    float
        tr_rhoA_2, purity for system A
    float
        tr_rhoB_2, purity for system B
    """
    nb_qubits = None

    # default circuit naming functions
    if circ_namesA is None:
        def circ_namesA(idx):
            return 'CrossFid' + f'{idx}'
    if circ_namesB is None:
        def circ_namesB(idx):
            return 'CrossFid' + f'{idx}'

    # make results access maps for speed
    resultsA_access_map = {
        res.header.name: idx for idx, res in enumerate(resultsA.results)
    }
    resultsB_access_map = {
        res.header.name: idx for idx, res in enumerate(resultsB.results)
    }

    # iterate over the different random unitaries
    tr_rhoA_rhoB = np.zeros(nb_random)
    tr_rhoA_2 = np.zeros(nb_random)
    tr_rhoB_2 = np.zeros(nb_random)
    for uidx in range(nb_random):

        # try to extract matching experiment data
        try:
            experiment_idx = resultsA_access_map[circ_namesA(uidx)]
            num_qubits_A, P_A_strings, P_A_counts = _unpack_experiment(
                experiment_idx, resultsA)
        except KeyError as missing_exp:
            raise ValueError('Cannot extract matching experiment data to'
                             + ' calculate cross-fidelity.') from missing_exp

        try:
            experiment_idx = resultsB_access_map[circ_namesB(uidx)]
            num_qubits_B, P_B_strings, P_B_counts = _unpack_experiment(
                experiment_idx, resultsB)
        except KeyError as missing_exp:
            raise ValueError('Cannot extract matching experiment data to'
                             + ' calculate cross-fidelity.') from missing_exp

        # use this to check number of qubits has been consistent
        # over all random unitaries
        if nb_qubits is None:
            # get the first dict key string and find its length
            nb_qubits = num_qubits_A
        if not nb_qubits == num_qubits_A:
            raise ValueError(
                'nb_qubits='+f'{nb_qubits}' + ', num_qubits_A='
                + f'{num_qubits_A}'
            )
        if not nb_qubits == num_qubits_B:
            raise ValueError(
                'nb_qubits='+f'{nb_qubits}' + ', num_qubits_B='
                + f'{num_qubits_B}'
            )

        if vectorise:
            cross_func = _vectorised_cross_correlation_fixed_u
            auto_func = _vectorised_auto_cross_correlation_fixed_u
        else:
            cross_func = _cross_correlation_fixed_u
            auto_func = _auto_cross_correlation_fixed_u

        tr_rhoA_rhoB[uidx] = cross_func(
            P_A_strings, P_A_counts,
            P_B_strings, P_B_counts,
            nb_qubits,
        )
        tr_rhoA_2[uidx] = auto_func(
            P_A_strings, P_A_counts, nb_qubits)
        tr_rhoB_2[uidx] = auto_func(
            P_B_strings, P_B_counts, nb_qubits)

    # normalisations
    tr_rhoA_rhoB = (2**nb_qubits)*tr_rhoA_rhoB
    tr_rhoA_2 = (2**nb_qubits)*tr_rhoA_2
    tr_rhoB_2 = (2**nb_qubits)*tr_rhoB_2

    return tr_rhoA_rhoB, tr_rhoA_2, tr_rhoB_2


def _hex_to_bin(hexstring):
    """
    Convert hexadecimal readouts (memory) to binary readouts.
    (Copied from qiskit source -- qiskit.result.postprocess)
    """
    return str(bin(int(hexstring, 16)))[2:]


def _pad_zeros(bitstring, memory_slots):
    """
    If the bitstring is truncated, pad extra zeros to make its length equal to
    memory_slots.
    (Copied from qiskit source -- qiskit.result.postprocess)
    """
    return format(int(bitstring, 2), '0{}b'.format(memory_slots))


def _cross_correlation_fixed_u(
    P_1_strings,
    P_1_counts,
    P_2_strings,
    P_2_counts,
    num_qubits,
):
    """
    Carries out the inner loop calculation of the Cross-Fidelity. In
    contrast to the paper, arxiv:1909.01282, it makes sense for us to
    make the sum over sA and sA' the inner loop. So this computes the
    sum over sA and sA' for fixed random U.

    Parameters
    ----------
    P_1_strings : tuple[str]
        List of measurement strings in hexidecimal for distribution 1
    P_1_counts : np.ndarray
        Counts histogram for the measurments on qubit 1
        P^{(1)}_U(s_A) = Tr[ U_A rho_1 U^dagger_A |s_A rangle langle s_A| ]
        where U is a fixed, randomly chosen unitary, and s_A is the set of
        strings (in corresponding order) given by P_1_strings
    P_2_strings : tuple[str]
        List of measurement strings in hexidecimal for distribution 2
    P_2_counts : np.ndarray
        Counts histogram for the measurments on qubit 2
        P^{(2)}_U(s_B) = Tr[ U_B rho_2 U^dagger_B |s_B rangle langle s_B| ]
        where U is a fixed, randomly chosen unitary, and s_B is the set of
        strings (in corresponding order) given by P_2_strings
    num_qubits : int
        Number of qubits measured

    Returns
    -------
    float
        Evaluation of the inner sum of the cross-fidelity
    """
    # normalise counts
    P_1_dist = P_1_counts / np.sum(P_1_counts)
    P_2_dist = P_2_counts / np.sum(P_2_counts)

    # iterate over the elements of the computational basis (that
    # appear in the measurement results)sublimes
    corr_fixed_u = 0
    for hex_sA, P_1_sA in zip(P_1_strings, P_1_dist):
        for hex_sAprime, P_2_sAprime in zip(P_2_strings, P_2_dist):

            # convert hexidecimal string to binary
            sA = _pad_zeros(_hex_to_bin(hex_sA), num_qubits)
            sAprime = _pad_zeros(_hex_to_bin(hex_sAprime), num_qubits)

            # add up contribution
            hamming_distance = int(
                len(sA)*sp.spatial.distance.hamming(list(sA), list(sAprime))
            )
            corr_fixed_u += (
                (-2)**(-hamming_distance) * P_1_sA*P_2_sAprime
            )

    return corr_fixed_u


def _auto_cross_correlation_fixed_u(P_1_strings, P_1_counts, num_qubits):
    """
    Carries out the inner loop purity calculation of arxiv:1909.01282 and
    arxiv:1801.00999, etc. In contrast to the two-source calculation above
    (_cross_correlation_fixed_u), in this case there is only one measured
    distribution of bit string probabilities so we need to take additional care
    to avoid estimator bias when computing cross-correlations.

    Parameters
    ----------
    P_1_strings : tuple[str]
        List of measurement strings in hexidecimal for distribution 1
    P_1_counts : np.ndarray
        Counts histogram for the measurments on qubit 1
        P^{(1)}_U(s_A) = Tr[ U_A rho_1 U^dagger_A |s_A rangle langle s_A| ]
        where U is a fixed, randomly chosen unitary, and s_A is the set of
        strings (in corresponding order) given by P_1_strings
    num_qubits : int
        Number of qubits measured

    Returns
    -------
    float
        Evaluation of the inner sum of the cross-fidelity
    """
    # normalise counts
    num_measurements = np.sum(P_1_counts)
    P_1_dist = P_1_counts / num_measurements

    # iterate over the elements of the computational basis (that
    # appear in the measurement results)sublimes
    corr_fixed_u = 0
    for hex_sA, P_1_sA in zip(P_1_strings, P_1_dist):
        for hex_sAprime, P_1_sAprime in zip(P_1_strings, P_1_dist):

            # convert hexidecimal string to binary
            sA = _pad_zeros(_hex_to_bin(hex_sA), num_qubits)
            sAprime = _pad_zeros(_hex_to_bin(hex_sAprime), num_qubits)

            if sA == sAprime:
                # bias corrected
                corr_fixed_u += (
                    P_1_sA * (num_measurements*P_1_sA - 1)
                    / (num_measurements - 1)
                )
            else:
                hamming_distance = int(
                    len(sA)*sp.spatial.distance.hamming(list(sA),
                                                        list(sAprime))
                )
                # bias corrected
                corr_fixed_u += (
                    (-2)**(-hamming_distance) * P_1_sA*P_1_sAprime
                    * num_measurements / (num_measurements - 1)
                )

    return corr_fixed_u


@functools.lru_cache(maxsize=1)
def _make_full_hamming_distance_matrix(num_qubits):
    """ """
    return np.count_nonzero(
        (
            np.array(
                [list(_pad_zeros(str(bin(val))[2:], num_qubits))
                 for val in range(2**num_qubits)]
            )[:, np.newaxis]
            != np.array(
                [list(_pad_zeros(str(bin(val))[2:], num_qubits))
                 for val in range(2**num_qubits)]
            )[np.newaxis, :]
        ),
        axis=2,
    )


def _make_hamming_distance_matrix(hexstrings_a, hexstrings_b, num_qubits):
    """ """
    full_distance_matrix = _make_full_hamming_distance_matrix(num_qubits)
    idx_a = [int(hexval, 16) for hexval in hexstrings_a]
    idx_b = [int(hexval, 16) for hexval in hexstrings_b]
    return full_distance_matrix[np.ix_(idx_a, idx_b)]


def _vectorised_cross_correlation_fixed_u(
    P_1_strings,
    P_1_counts,
    P_2_strings,
    P_2_counts,
    num_qubits,
):
    """
    Carries out the inner loop calculation of the Cross-Fidelity. In
    contrast to the paper, arxiv:1909.01282, it makes sense for us to
    make the sum over sA and sA' the inner loop. So this computes the
    sum over sA and sA' for fixed random U.

    Vectorised calculation should be slightly faster.

    Parameters
    ----------
    P_1_strings : tuple[str]
        List of measurement strings in hexidecimal for distribution 1
    P_1_counts : np.ndarray
        Counts histogram for the measurments on qubit 1
        P^{(1)}_U(s_A) = Tr[ U_A rho_1 U^dagger_A |s_A rangle langle s_A| ]
        where U is a fixed, randomly chosen unitary, and s_A is the set of
        strings (in corresponding order) given by P_1_strings
    P_2_strings : tuple[str]
        List of measurement strings in hexidecimal for distribution 2
    P_2_counts : np.ndarray
        Counts histogram for the measurments on qubit 2
        P^{(2)}_U(s_B) = Tr[ U_B rho_2 U^dagger_B |s_B rangle langle s_B| ]
        where U is a fixed, randomly chosen unitary, and s_B is the set of
        strings (in corresponding order) given by P_2_strings
    num_qubits : int
        Number of qubits measured

    Returns
    -------
    float
        Evaluation of the inner sum of the cross-fidelity
    """
    # normalise counts
    P_1_dist = P_1_counts / np.sum(P_1_counts)
    P_2_dist = P_2_counts / np.sum(P_2_counts)

    hamming_distances = _make_hamming_distance_matrix(
        P_1_strings, P_2_strings, num_qubits)

    return np.sum(
        (-2.) ** (-1*hamming_distances)
        * P_1_dist[:, np.newaxis]
        * P_2_dist[np.newaxis, :]
    )


def _vectorised_auto_cross_correlation_fixed_u(
    P_1_strings,
    P_1_counts,
    num_qubits,
):
    """
    Carries out the inner loop purity calculation of arxiv:1909.01282 and
    arxiv:1801.00999, etc. In contrast to the two-source calculation above
    (_cross_correlation_fixed_u), in this case there is only one measured
    distribution of bit string probabilities so we need to take additional care
    to avoid estimator bias when computing cross-correlations.

    Vectorised calculation should be slightly faster.

    Parameters
    ----------
    P_1_strings : tuple[str]
        List of measurement strings in hexidecimal for distribution 1
    P_1_counts : np.ndarray
        Counts histogram for the measurments on qubit 1
        P^{(1)}_U(s_A) = Tr[ U_A rho_1 U^dagger_A |s_A rangle langle s_A| ]
        where U is a fixed, randomly chosen unitary, and s_A is the set of
        strings (in corresponding order) given by P_1_strings
    num_qubits : int
        Number of qubits measured

    Returns
    -------
    float
        Evaluation of the inner sum of the cross-fidelity
    """
    # normalise counts
    num_measurements = np.sum(P_1_counts)
    P_1_dist = P_1_counts / num_measurements

    hamming_distances = _make_hamming_distance_matrix(
        P_1_strings, P_1_strings, num_qubits)

    vectorised_sum = (
        (-2.) ** (-1*hamming_distances)
        * P_1_dist[:, np.newaxis]
        * P_1_dist[np.newaxis, :]
    )

    # correct bias
    vectorised_sum = (
        vectorised_sum * num_measurements
        - np.diag(P_1_dist)
    ) / (num_measurements - 1)

    return np.sum(vectorised_sum)
