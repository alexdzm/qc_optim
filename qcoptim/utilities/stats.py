"""
"""

import numpy as np
from numpy.random import default_rng


def resample_histogram(counts, num_resamples, random_seed=None):
    """
    Resample a probability distribution specified as a set of event frequencies
    (i.e. counts of a histogram) to generate a new set of frequencies for those
    events. As `num_resamples` becomes large the (normalised) histogram
    returned by this function will converge to `counts`.

    Parameters
    ----------
    counts : numpy.ndarray
        Distribution to resample, specified as histogram counts. These can be
        normalised (sum to one) or unnormalised
    num_resamples : int
        Number of samples to draw from the distribution `counts` to generate a
        resampled histogram
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    resampled_histogram : numpy.ndarray
        New histogram generated from resampling
    """
    rng = default_rng(seed=random_seed)
    cdf = np.cumsum(counts)

    # draw random integers between 0 and sum(counts)
    ints_draw = cdf[-1] * rng.random(size=num_resamples)

    # identify which bin each random int falls into
    resampled_bin_assignments = (
        ints_draw[:, np.newaxis] < cdf[np.newaxis, :]
    ).argmax(axis=1)

    # calculate frequencies that random ints fell into each bin
    unique, new_counts = np.unique(resampled_bin_assignments,
                                   return_counts=True)

    # deal with empty bins in the resampled histogram and format output
    tmp = dict(zip(unique, new_counts))
    new_hist = []
    for histbin in range(len(counts)):
        if histbin not in tmp.keys():
            new_hist.append(0)
        else:
            new_hist.append(tmp[histbin])
    return np.array(new_hist)


def bootstrap_resample(stat_func, observations, num_bootstraps,
                       return_dist=False, random_seed=None, ):
    """
    Calculate the boostrap mean and standard-error of `stat_func` applied to
    `observations` dataset. Optionally return the list of resampled
    values of the estimator, instead of the standard error.

    Parameters
    ----------
    stat_func : Callable
        Function to evaluate on each bootstrap resample e.g. np.std
    observations : np.ndarray
        Data to resample
    num_bootstraps : int
        Number of bootstrap resamples to perform
    return_dist : boolean, default False
        If True, return the list of resampled values of the estimator instead
        of the standard error (see Returns)
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    mean_estimate : float
    standard_error OR resampled_values : float OR list[float]
        If `return_dist=True` returns list of resampled values, else returns
        standard error
    """
    num_data = observations.size
    rng = default_rng(seed=random_seed)

    try:
        # try to vectorise the bootstrapping, unless the size of the resulting
        # array will be too large (134217728 is approx the size of a 1GB
        # float64 array)
        if num_bootstraps * num_data > 134217728:
            raise TypeError

        # vectorisation will also fail if `stat_func` does not have an `axis`
        # kwarg, which will raise a TypeError here
        resample_indexes = rng.integers(
            0, num_data, size=(num_bootstraps, num_data))
        resampled_estimator = stat_func(
            observations[resample_indexes], axis=1
            )
    except TypeError:
        # more memory safe but much slower
        resampled_estimator = np.zeros(num_bootstraps)
        for boot in range(num_bootstraps):
            resample_indexes = rng.integers(0, num_data, size=num_data)
            resampled_estimator[boot] = stat_func(
                observations[resample_indexes])

    if return_dist:
        # bootstrap estimate and list of resampled values
        return (
            np.mean(resampled_estimator),
            resampled_estimator
        )
    # bootstrap estimate and standard deviation
    return (
        np.mean(resampled_estimator),
        np.std(resampled_estimator)
    )
