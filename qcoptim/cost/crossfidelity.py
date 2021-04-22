"""
Cross-fidelity cost classes and functions
"""

import sys

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
        instance,
        nb_random=None,
        seed=None,
        comparison_results=None,
        prefixA='CrossFid',
        prefixB='CrossFid',
        rand_meas_handler=None,
    ):
        """
        Parameters
        ----------
        ansatz : object implementing AnsatzInterface
            The ansatz object that this cost can be optimsed over
        instance : qiskit quantum instance
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
        prefixA : str, optional
            Prefix string to use on circuits generate to characterise A system.
        prefixB : str, optional
            Prefix string to use to extract system B's results from
            comparison_results.
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
        # check if comparison_results contains the crossfidelity_metadata
        # tags and if it does compare them, if these comparisons fail then
        # crash, if the crossfidelity_metadata is missing issue a warning
        if results is not None:
            if not isinstance(results, dict):
                results = results.to_dict()

            comparison_metadata = None
            try:
                comparison_metadata = results['crossfidelity_metadata']
            except KeyError:
                print(
                    'Warning, input results dictionary does not contain'
                    + ' crossfidelity_metadata and so we cannot confirm that'
                    + ' the results are compatible. If the input results'
                    + ' object was collecting by this class consider using'
                    + ' the tag_results_metadata method to add the'
                    + ' crossfidelity_metadata.', file=sys.stderr
                )
            if comparison_metadata is not None:
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

        self._comparison_results = results

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
        return self.evaluate_cost_and_std(results, name=name, **kwargs)[0]

    def evaluate_cost_and_std(
        self,
        results,
        name='',
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
        float
            Evaluated cross-fidelity
        float
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

        # convert comparison_results back to qiskit results obj, so we can
        # use `get_counts` method
        comparison_results = Result.from_dict(self._comparison_results)

        # circuit naming functions
        def circ_namesA(idx):
            return name + self._rand_meas_handler.circ_name(idx)
        def circ_namesB(idx):
            return self._prefixB + f'{idx}'

        (dist_tr_rhoA_rhoB,
         dist_tr_rhoA_2,
         dist_tr_rhoB_2) = _crossfidelity_fixed_u(
         results, comparison_results, self.nb_random,
         circ_namesA=circ_namesA, circ_namesB=circ_namesB,
        )

        # bootstrap resample for means and std-errs
        tr_rhoA_rhoB, tr_rhoA_rhoB_err = bootstrap_resample(
            np.mean, dist_tr_rhoA_rhoB, 1000,
        )
        tr_rhoA_2, tr_rhoA_2_err = bootstrap_resample(
            np.mean, dist_tr_rhoA_2, 1000,
        )
        tr_rhoB_2, tr_rhoB_2_err = bootstrap_resample(
            np.mean, dist_tr_rhoB_2, 1000,
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


def _crossfidelity_fixed_u(
    resultsA,
    resultsB,
    nb_random,
    circ_namesA=None,
    circ_namesB=None,
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

    # iterate over the different random unitaries
    tr_rhoA_rhoB = np.zeros(nb_random)
    tr_rhoA_2 = np.zeros(nb_random)
    tr_rhoB_2 = np.zeros(nb_random)
    for uidx in range(nb_random):

        # try to extract matching experiment data
        try:
            countsdict_rhoA_fixedU = resultsA.get_counts(circ_namesA(uidx))
        except QiskitError as missing_exp:
            raise ValueError('Cannot extract matching experiment data to'
                             + ' calculate cross-fidelity.') from missing_exp

        try:
            countsdict_rhoB_fixedU = resultsB.get_counts(circ_namesB(uidx))
        except QiskitError as missing_exp:
            raise ValueError('Cannot extract matching experiment data to'
                             + ' calculate cross-fidelity.') from missing_exp

        # normalise counts dict to give empirical probability dists
        P_rhoA_fixedU = {
            k: v/sum(countsdict_rhoA_fixedU.values())
            for k, v in countsdict_rhoA_fixedU.items()
        }
        P_rhoB_fixedU = {
            k: v/sum(countsdict_rhoB_fixedU.values())
            for k, v in countsdict_rhoB_fixedU.items()
        }

        # use this to check number of qubits has been consistent
        # over all random unitaries
        if nb_qubits is None:
            # get the first dict key string and find its length
            nb_qubits = len(list(P_rhoA_fixedU.keys())[0])
        if not nb_qubits == len(list(P_rhoA_fixedU.keys())[0]):
            raise ValueError(
                'nb_qubits='+f'{nb_qubits}' + ', P_rhoA_fixedU.keys()='
                + f'{P_rhoA_fixedU.keys()}'
            )
        if not nb_qubits == len(list(P_rhoB_fixedU.keys())[0]):
            raise ValueError(
                'nb_qubits='+f'{nb_qubits}' + ', P_rhoB_fixedU.keys()='
                + f'{P_rhoB_fixedU.keys()}'
            )

        tr_rhoA_rhoB[uidx] = _correlation_fixed_u(P_rhoA_fixedU, P_rhoB_fixedU)
        tr_rhoA_2[uidx] = _correlation_fixed_u(P_rhoA_fixedU, P_rhoA_fixedU)
        tr_rhoB_2[uidx] = _correlation_fixed_u(P_rhoB_fixedU, P_rhoB_fixedU)

    # normalisations
    tr_rhoA_rhoB = (2**nb_qubits)*tr_rhoA_rhoB
    tr_rhoA_2 = (2**nb_qubits)*tr_rhoA_2
    tr_rhoB_2 = (2**nb_qubits)*tr_rhoB_2

    return tr_rhoA_rhoB, tr_rhoA_2, tr_rhoB_2


def _correlation_fixed_u(P_1, P_2):
    """
    Carries out the inner loop calculation of the Cross-Fidelity. In
    contrast to the paper, arxiv:1909.01282, it makes sense for us to
    make the sum over sA and sA' the inner loop. So this computes the
    sum over sA and sA' for fixed random U.

    Parameters
    ----------
    P_1 : dict (normalised counts dictionary)
        The empirical distribution for the measurments on qubit 1
        P^{(1)}_U(s_A) = Tr[ U_A rho_1 U^dagger_A |s_A rangle langle s_A| ]
        where U is a fixed, randomly chosen unitary, and s_A is all possible
        binary strings in the computational basis
    P_2 : dict (normalised counts dictionary)
        Same for qubit 2.

    Returns
    -------
    float
        Evaluation of the inner sum of the cross-fidelity
    """
    # iterate over the elements of the computational basis (that
    # appear in the measurement results)sublimes
    corr_fixed_u = 0
    for sA, P_1_sA in P_1.items():
        for sAprime, P_2_sAprime in P_2.items():

            # add up contribution
            hamming_distance = int(
                len(sA)*sp.spatial.distance.hamming(list(sA), list(sAprime))
            )
            corr_fixed_u += (
                (-2)**(-hamming_distance) * P_1_sA*P_2_sAprime
            )

    return corr_fixed_u
